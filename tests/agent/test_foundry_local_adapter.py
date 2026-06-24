"""Tests for agent.foundry_local_adapter (Foundry Local SDK wrapper).

The SDK isn't installed in CI, so these tests inject fake SDK modules that mimic
both the current (``foundry_local_sdk`` singleton + web service) and legacy
(``foundry_local`` constructor) surfaces.
"""

from __future__ import annotations

import types

import pytest

from agent import foundry_local_adapter as fl


@pytest.fixture(autouse=True)
def _reset_adapter_state():
    fl.reset_state()
    yield
    fl.reset_state()


# ---------------------------------------------------------------------------
# Fake "current" (1.x) SDK
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, alias, mid, *, cached=True, loaded=False):
        self.alias = alias
        self.id = mid
        self.is_cached = cached
        self.is_loaded = loaded
        self.download_calls = 0
        self.load_calls = 0

    def download(self, cb=None):
        self.download_calls += 1
        self.is_cached = True
        if cb is not None:
            cb(100.0)

    def load(self):
        self.load_calls += 1
        self.is_loaded = True


class _FakeCatalog:
    def __init__(self, models):
        self.models = list(models)
        self._by_ref = {}
        for m in models:
            self._by_ref[m.alias] = m
            self._by_ref.setdefault(m.id, m)

    def list_models(self):
        return list(self.models)

    def get_model(self, ref):
        return self._by_ref.get(ref)


def _make_current_sdk(models, *, urls=None):
    catalog = _FakeCatalog(models)
    state = {"web_started": 0, "eps": 0, "config": None}

    class FakeManager:
        instance = None

        def __init__(self):
            self.catalog = catalog
            self.urls = list(urls or ["http://127.0.0.1:55001"])

        @classmethod
        def initialize(cls, config):
            state["config"] = config
            cls.instance = cls()

        def start_web_service(self):
            state["web_started"] += 1

        def download_and_register_eps(self, progress_callback=None):
            state["eps"] += 1
            if progress_callback is not None:
                progress_callback("CPUExecutionProvider", 100.0)

    mod = types.SimpleNamespace(
        Configuration=lambda **kw: types.SimpleNamespace(**kw),
        FoundryLocalManager=FakeManager,
    )
    return mod, state, catalog


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def test_list_models_current(monkeypatch):
    models = [
        _FakeModel("qwen2.5-0.5b", "qwen2.5-0.5b-instruct-generic-cpu"),
        _FakeModel("phi-3.5-mini", "phi-3.5-mini-instruct-cuda-gpu", loaded=True),
    ]
    mod, _state, _catalog = _make_current_sdk(models)
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "current"))

    out = fl.list_models()
    aliases = [m.alias for m in out]
    assert aliases == ["qwen2.5-0.5b", "phi-3.5-mini"]
    assert out[0].id == "qwen2.5-0.5b-instruct-generic-cpu"
    assert out[1].loaded is True


# ---------------------------------------------------------------------------
# provision
# ---------------------------------------------------------------------------


def test_provision_loads_cached_model_and_returns_endpoint(monkeypatch):
    model = _FakeModel("qwen2.5-0.5b", "qwen2.5-0.5b-instruct-generic-cpu", cached=True, loaded=False)
    mod, state, _catalog = _make_current_sdk([model], urls=["http://127.0.0.1:5273"])
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "current"))

    rt = fl.provision("qwen2.5-0.5b")
    # OpenAI-compatible base URL with /v1 appended exactly once.
    assert rt.base_url == "http://127.0.0.1:5273/v1"
    # Wire model id is the concrete variant, not the alias.
    assert rt.model_id == "qwen2.5-0.5b-instruct-generic-cpu"
    assert rt.alias == "qwen2.5-0.5b"
    assert rt.api_key == fl.LOCAL_PLACEHOLDER_API_KEY
    # Cached model: loaded but not (re)downloaded; web service started.
    assert model.load_calls == 1
    assert model.download_calls == 0
    assert state["web_started"] == 1


def test_provision_is_cached_per_process(monkeypatch):
    model = _FakeModel("qwen2.5-0.5b", "qwen2.5-0.5b-cpu", cached=True)
    mod, state, _catalog = _make_current_sdk([model])
    calls = {"import": 0}

    def _imp(allow_install=False):
        calls["import"] += 1
        return mod, "current"

    monkeypatch.setattr(fl, "_import_sdk", _imp)

    rt1 = fl.provision("qwen2.5-0.5b")
    rt2 = fl.provision("qwen2.5-0.5b")
    assert rt1 == rt2
    # Second call served from cache: SDK not re-imported, model not re-loaded.
    assert calls["import"] == 1
    assert model.load_calls == 1


def test_provision_uncached_without_download_raises(monkeypatch):
    model = _FakeModel("qwen2.5-0.5b", "qwen2.5-0.5b-cpu", cached=False)
    mod, _state, _catalog = _make_current_sdk([model])
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "current"))

    with pytest.raises(fl.FoundryLocalError) as exc:
        fl.provision("qwen2.5-0.5b", allow_download=False)
    assert "not downloaded" in str(exc.value).lower()
    assert model.load_calls == 0


def test_provision_downloads_when_allowed(monkeypatch):
    model = _FakeModel("qwen2.5-0.5b", "qwen2.5-0.5b-cpu", cached=False)
    mod, state, _catalog = _make_current_sdk([model])
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=True: (mod, "current"))

    seen = []
    rt = fl.provision(
        "qwen2.5-0.5b",
        allow_download=True,
        allow_install=True,
        on_progress=lambda stage, pct: seen.append((stage, pct)),
    )
    assert rt.model_id == "qwen2.5-0.5b-cpu"
    assert model.download_calls == 1
    assert model.load_calls == 1
    assert state["eps"] == 1  # execution providers registered before download
    # progress surfaced for both EP registration and model download
    assert any(s.startswith("ep:") for s, _ in seen)
    assert any(s == "download" for s, _ in seen)


def test_provision_unknown_model_raises(monkeypatch):
    mod, _state, _catalog = _make_current_sdk([_FakeModel("a", "a-cpu")])
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "current"))
    with pytest.raises(fl.FoundryLocalError):
        fl.provision("does-not-exist")


def test_provision_empty_ref_raises(monkeypatch):
    # Should fail before importing the SDK.
    monkeypatch.setattr(
        fl, "_import_sdk",
        lambda allow_install=False: (_ for _ in ()).throw(AssertionError("should not import")),
    )
    with pytest.raises(fl.FoundryLocalError):
        fl.provision("   ")


# ---------------------------------------------------------------------------
# legacy (0.x) SDK
# ---------------------------------------------------------------------------


def test_provision_legacy_surface(monkeypatch):
    class LegacyManager:
        def __init__(self, ref):
            self.ref = ref
            self.endpoint = "http://localhost:1234/v1"
            self.api_key = ""  # local: empty

        def get_model_info(self, ref):
            return types.SimpleNamespace(id=f"{ref}-generic-cpu")

    mod = types.SimpleNamespace(FoundryLocalManager=LegacyManager)
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "legacy"))

    rt = fl.provision("phi-3.5-mini")
    assert rt.base_url == "http://localhost:1234/v1"
    assert rt.model_id == "phi-3.5-mini-generic-cpu"
    # Empty api_key falls back to the placeholder.
    assert rt.api_key == fl.LOCAL_PLACEHOLDER_API_KEY


def test_endpoint_without_v1_gets_suffix(monkeypatch):
    class LegacyManager:
        def __init__(self, ref):
            self.endpoint = "http://localhost:1234"
            self.api_key = "k"

        def get_model_info(self, ref):
            return types.SimpleNamespace(id=ref)

    mod = types.SimpleNamespace(FoundryLocalManager=LegacyManager)
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "legacy"))
    rt = fl.provision("m")
    assert rt.base_url == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# error normalization
# ---------------------------------------------------------------------------


def test_sdk_runtime_error_is_wrapped(monkeypatch):
    class BoomManager:
        instance = None

        @classmethod
        def initialize(cls, config):
            raise RuntimeError("service refused to start")

    mod = types.SimpleNamespace(
        Configuration=lambda **kw: types.SimpleNamespace(**kw),
        FoundryLocalManager=BoomManager,
    )
    monkeypatch.setattr(fl, "_import_sdk", lambda allow_install=False: (mod, "current"))
    with pytest.raises(fl.FoundryLocalError) as exc:
        fl.provision("m")
    assert "service refused to start" in str(exc.value)
