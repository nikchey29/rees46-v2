from rees46.experiments.config import load_experiment_config


def test_modeling_config_is_valid() -> None:
    config = load_experiment_config()

    assert config.dev.max_eval_users < config.full.max_eval_users
    assert 10 in config.k_values
    assert config.full.sequential_epochs >= config.dev.sequential_epochs
