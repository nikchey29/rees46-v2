from rees46 import __version__
from rees46.health import project_name


def test_project_name() -> None:
    assert project_name() == "REES46 V2"


def test_project_version() -> None:
    assert __version__ == "0.1.0"
