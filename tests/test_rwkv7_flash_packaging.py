# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement

from fla.ops.rwkv7.backends.flash_rwkv import FLASH_RWKV_SOURCE_REVISION
from scripts.build_packages import build_split_packages

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
FLASH_RWKV_EXTRA = "flash-rwkv"
FLASH_RWKV_REPOSITORY = "git+https://github.com/rwkv-rs/FlashRWKV.git"


def _read_pyproject(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def _extra_requirement(pyproject: dict, extra: str) -> Requirement:
    dependencies = pyproject["project"]["optional-dependencies"][extra]
    assert len(dependencies) == 1
    return Requirement(dependencies[0])


def _assert_flash_rwkv_requirement(requirement: Requirement) -> None:
    assert requirement.name == FLASH_RWKV_EXTRA
    assert requirement.specifier == ""
    assert requirement.marker is None
    assert requirement.url == f"{FLASH_RWKV_REPOSITORY}@{FLASH_RWKV_SOURCE_REVISION}"
    assert requirement.url.count(FLASH_RWKV_SOURCE_REVISION) == 1


def _build_project_wheel(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    shutil.copytree(
        ROOT / "fla",
        source_dir / "fla",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source_dir / name)

    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    build_script = """
import os
import sys

from setuptools.build_meta import build_wheel

os.chdir(sys.argv[1])
print(build_wheel(sys.argv[2]))
""".strip()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            build_script,
            str(source_dir),
            str(wheel_dir),
        ],
        check=False,
        cwd=tmp_path,
        env=env,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    wheels = list(wheel_dir.glob("flash_linear_attention-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _read_wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1
        return BytesParser().parsebytes(archive.read(metadata_files[0]))


def test_flash_rwkv_extra_pins_backend_source_revision() -> None:
    requirement = _extra_requirement(_read_pyproject(ROOT / "pyproject.toml"), FLASH_RWKV_EXTRA)
    _assert_flash_rwkv_requirement(requirement)


def test_flash_rwkv_extra_is_preserved_in_wheel_metadata(tmp_path: Path) -> None:
    metadata = _read_wheel_metadata(_build_project_wheel(tmp_path))
    assert (metadata.get_all("Provides-Extra") or []).count(FLASH_RWKV_EXTRA) == 1

    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist") or []]
    matching = [
        requirement
        for requirement in requirements
        if requirement.name == FLASH_RWKV_EXTRA
        and requirement.marker is not None
        and requirement.marker.evaluate({"extra": FLASH_RWKV_EXTRA})
    ]
    assert len(matching) == 1
    requirement = matching[0]
    assert requirement.url == f"{FLASH_RWKV_REPOSITORY}@{FLASH_RWKV_SOURCE_REVISION}"
    assert not requirement.marker.evaluate({"extra": "cuda"})


def test_split_packages_route_flash_rwkv_extra_through_core(tmp_path: Path) -> None:
    output_dir, version = build_split_packages(tmp_path / "split")
    core_pyproject = _read_pyproject(output_dir / "fla-core" / "pyproject.toml")
    extension_pyproject = _read_pyproject(output_dir / "flash-linear-attention" / "pyproject.toml")

    _assert_flash_rwkv_requirement(_extra_requirement(core_pyproject, FLASH_RWKV_EXTRA))
    forwarded = _extra_requirement(extension_pyproject, FLASH_RWKV_EXTRA)
    assert forwarded.name == "fla-core"
    assert forwarded.extras == {FLASH_RWKV_EXTRA}
    assert str(forwarded.specifier) == f"=={version}"
    assert forwarded.url is None
    assert forwarded.marker is None
