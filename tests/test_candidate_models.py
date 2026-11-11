"""Issues #14-#18: newer models of the paper's providers are opt-in, isolated, and extendable."""
import importlib

import config
import main as pipeline_main


def test_candidates_never_in_default_run():
    default = set(config.all_models())
    for prov in ("openai", "gemini", "mistral"):
        assert not set(config.EXTRA_MODELS[prov]) & default
    assert pipeline_main._resolve_models(None, config.all_models(), config.selectable_models()) == config.all_models()


def test_candidates_resolve_to_right_provider_and_own_folder():
    assert config.model_provider("gpt-5.6-sol") == "openai"
    assert config.model_provider("gemini-3.6-flash") == "gemini"
    assert config.model_output_dir("gpt-5.6-sol") == "openai_gpt_5_6_sol"
    assert config.model_output_dir("gemini-3.6-flash") == "gemini_gemini_3_6_flash"
    assert config.model_output_dir("gpt-5.1") == "openai"                       # the paper snapshot is untouched


def test_new_aliases():
    sel = config.selectable_models()
    assert pipeline_main._resolve_models(["openai-new"], config.all_models(), sel) == config.EXTRA_MODELS["openai"]
    assert pipeline_main._resolve_models(["gemini-new"], config.all_models(), sel) == config.EXTRA_MODELS["gemini"]


def test_env_can_add_models(monkeypatch):
    monkeypatch.setenv("OPENAI_EXTRA_MODELS", "my-gpt-x, other-gpt")
    monkeypatch.setenv("MISTRAL_EXTRA_MODELS", "mistral-medium-x")
    try:
        importlib.reload(config)
        assert {"my-gpt-x", "other-gpt"} <= set(config.EXTRA_MODELS["openai"])
        assert config.model_provider("mistral-medium-x") == "mistral"
        assert config.model_output_dir("mistral-medium-x") == "mistral_mistral_medium_x"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_every_selectable_model_has_unique_results_folder_and_known_provider():
    models = config.selectable_models()
    assert len(models) == len(set(models))
    dirs = [config.model_output_dir(m) for m in models]
    assert len(dirs) == len(set(dirs))
    assert all(config.model_provider(m) in config.PROVIDER_KEYS for m in models)
