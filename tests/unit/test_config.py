from rees46.runtime.config import PROJECT_ROOT, Profile, load_config


def test_dev_profile_loads() -> None:
    config = load_config(Profile.DEV)

    assert config.profile is Profile.DEV
    assert config.project.name == "REES46 V2"
    assert config.project.random_seed == 42
    assert config.logging.level == "DEBUG"
    assert config.logging.json_output is False


def test_full_profile_loads() -> None:
    config = load_config(Profile.FULL)

    assert config.profile is Profile.FULL
    assert config.logging.level == "INFO"
    assert config.logging.json_output is True


def test_project_paths_are_absolute() -> None:
    config = load_config(Profile.DEV)

    assert config.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert config.paths.bronze == PROJECT_ROOT / "data" / "bronze"
    assert config.paths.silver == PROJECT_ROOT / "data" / "silver"
    assert config.paths.gold == PROJECT_ROOT / "data" / "gold"

    assert config.paths.raw.is_absolute()
