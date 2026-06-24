"""Provider registration + runtime resolution for the foundry-local provider."""

from __future__ import annotations

import pytest

from hermes_cli import providers as P
from hermes_cli import runtime_provider as RP
from hermes_cli.auth import AuthError
from agent.foundry_local_adapter import FoundryLocalError, FoundryRuntime


# ---------------------------------------------------------------------------
# Provider identity / overlay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias",
    ["foundry-local", "foundrylocal", "foundry_local", "azure-foundry-local", "azure_foundry_local"],
)
def test_aliases_normalize_to_foundry_local(alias):
    assert P.normalize_provider(alias) == "foundry-local"


def test_overlay_is_openai_chat_local():
    pdef = P.get_provider("foundry-local")
    assert pdef is not None
    assert pdef.transport == "openai_chat"
    assert pdef.base_url_env_var == "FOUNDRY_LOCAL_BASE_URL"


def test_label():
    assert P.get_label("foundry-local") == "Foundry Local"


def test_api_mode_is_chat_completions():
    assert P.determine_api_mode("foundry-local") == "chat_completions"


def test_not_confused_with_cloud_azure_foundry():
    # The on-device provider must stay distinct from the cloud azure-foundry one.
    assert P.normalize_provider("azure-foundry") == "azure-foundry"
    assert RP._is_foundry_local("azure-foundry") is False
    assert RP._is_foundry_local("foundry-local") is True
    assert RP._is_foundry_local("azure-foundry-local") is True


def test_in_canonical_providers():
    from hermes_cli.models import CANONICAL_PROVIDERS

    slugs = {p.slug for p in CANONICAL_PROVIDERS}
    assert "foundry-local" in slugs


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------


@pytest.fixture
def _fl_config(monkeypatch):
    cfg = {"provider": "foundry-local", "default": "qwen2.5-0.5b"}
    monkeypatch.setattr(RP, "_get_model_config", lambda: dict(cfg))
    return cfg


def _patch_provision(monkeypatch, runtime=None, exc=None):
    calls = {}

    def _fake_provision(model_ref, **kwargs):
        calls["model_ref"] = model_ref
        calls["kwargs"] = kwargs
        if exc is not None:
            raise exc
        return runtime

    import agent.foundry_local_adapter as fl_mod

    monkeypatch.setattr(fl_mod, "provision", _fake_provision)
    return calls


def test_resolve_runtime_returns_dynamic_endpoint_and_concrete_model(monkeypatch, _fl_config):
    rt = FoundryRuntime(
        base_url="http://127.0.0.1:5273/v1",
        api_key="none",
        model_id="qwen2.5-0.5b-instruct-generic-cpu",
        alias="qwen2.5-0.5b",
    )
    calls = _patch_provision(monkeypatch, runtime=rt)

    resolved = RP.resolve_runtime_provider(requested="foundry-local")
    assert resolved["provider"] == "foundry-local"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://127.0.0.1:5273/v1"
    assert resolved["api_key"] == "none"
    # Concrete wire model id is surfaced for consumers to apply.
    assert resolved["model"] == "qwen2.5-0.5b-instruct-generic-cpu"
    assert resolved["source"] == "foundry-local-sdk"
    # The portable alias from config is what gets provisioned.
    assert calls["model_ref"] == "qwen2.5-0.5b"
    # Runtime resolution must never trigger a download.
    assert calls["kwargs"].get("allow_download") is False


def test_resolve_runtime_via_alias(monkeypatch, _fl_config):
    rt = FoundryRuntime("http://127.0.0.1:9/v1", "none", "qwen2.5-0.5b-cpu", "qwen2.5-0.5b")
    _patch_provision(monkeypatch, runtime=rt)
    resolved = RP.resolve_runtime_provider(requested="azure-foundry-local")
    assert resolved["provider"] == "foundry-local"
    assert resolved["base_url"] == "http://127.0.0.1:9/v1"


def test_target_model_overrides_config_default(monkeypatch, _fl_config):
    rt = FoundryRuntime("http://127.0.0.1:9/v1", "none", "phi-3.5-mini-cpu", "phi-3.5-mini")
    calls = _patch_provision(monkeypatch, runtime=rt)
    RP.resolve_runtime_provider(requested="foundry-local", target_model="phi-3.5-mini")
    assert calls["model_ref"] == "phi-3.5-mini"


def test_resolve_runtime_error_becomes_auth_error(monkeypatch, _fl_config):
    _patch_provision(monkeypatch, exc=FoundryLocalError("model is not downloaded yet"))
    with pytest.raises(AuthError) as exc:
        RP.resolve_runtime_provider(requested="foundry-local")
    assert "not downloaded" in str(exc.value).lower()


def test_missing_model_raises_auth_error(monkeypatch):
    monkeypatch.setattr(RP, "_get_model_config", lambda: {"provider": "foundry-local", "default": ""})
    with pytest.raises(AuthError):
        RP.resolve_runtime_provider(requested="foundry-local")
