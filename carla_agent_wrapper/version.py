from importlib import metadata
from pathlib import Path

import tomli

_DISTRIBUTION_NAME = "carla-agent-wrapper"


def wrapper_version() -> str:
    try:
        version = metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as pyproject_file:
            version = tomli.load(pyproject_file)["project"]["version"]

    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Unable to determine {_DISTRIBUTION_NAME} package version")
    return version
