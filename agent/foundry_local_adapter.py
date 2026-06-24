"""Microsoft Foundry Local adapter — on-device LLM provisioning + connect.

Foundry Local (https://learn.microsoft.com/azure/foundry-local) is Microsoft's
on-device inference runtime: it downloads hardware-optimized ONNX model variants
(CPU / CUDA / NPU / Vitis / QNN / OpenVINO) and exposes an OpenAI-compatible REST
endpoint on a locally-bound port. It is the same shape as LM Studio / Ollama —
a local OpenAI-compatible server — so Hermes drives inference through the
existing ``chat_completions`` transport. This adapter only handles the two
things the SDK is needed for: **provisioning** (start the service, download +
load the chosen model) and **connect** (discover the dynamic endpoint URL).

This is distinct from the ``azure-foundry`` provider, which targets the *cloud*
Azure AI Foundry service (remote endpoint + Entra ID / API key). Foundry Local
is fully on-device with no Azure subscription, matching Hermes' weak-hardware
single-model focus.

Architecture mirrors ``agent/azure_identity_adapter.py`` and
``agent/bedrock_adapter.py``:

* **Lazy import.** ``foundry-local-sdk`` (and its heavy ONNX Runtime tree) is
  only imported when the user actually selects the ``foundry-local`` provider.
  Everyone else never pays the import/install cost.
* **Two SDK generations supported.** The current SDK (``foundry-local-sdk`` 1.x,
  import ``foundry_local_sdk``) uses a singleton manager + an explicit web
  service (``FoundryLocalManager.initialize`` → ``.instance`` →
  ``start_web_service`` → ``manager.urls``). The legacy SDK (0.x, import
  ``foundry_local``) used ``FoundryLocalManager(alias)`` with ``.endpoint`` /
  ``.api_key`` / ``.get_model_info``. We feature-detect and support both so the
  integration keeps working across the SDK's transition.
* **Process-level caching.** Once a model is provisioned its endpoint is cached,
  so the per-turn runtime resolver (``hermes_cli.runtime_provider``) is cheap on
  every call after the first — important on weak hardware where each turn is
  precious.

Reference (Python): https://learn.microsoft.com/azure/foundry-local/how-to/how-to-integrate-with-inference-sdks?pivots=programming-language-python

Requires: ``foundry-local-sdk`` (optional dependency — only needed when the
``foundry-local`` provider is selected). Inference itself uses the core
``openai`` SDK via the chat_completions transport, not this package.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-install feature key — see tools/lazy_deps.py (provider.foundry_local).
_FOUNDRY_LOCAL_FEATURE = "provider.foundry_local"

# Local servers don't require auth; the OpenAI SDK still wants a non-empty
# api_key string, so we hand it a harmless placeholder (matches Microsoft's
# documented "none" / "notneeded" samples and Hermes' own local-endpoint
# placeholder in cli_agent_setup_mixin).
LOCAL_PLACEHOLDER_API_KEY = "none"

# App name reported to the Foundry Local service (shows up in its logs).
_APP_NAME = "hermes-agent"

# Progress callback: (stage, percent) where stage is e.g. "download" or
# "ep:CUDAExecutionProvider" and percent is 0..100 (may be -1 when unknown).
ProgressCallback = Callable[[str, float], None]

# ── Module state (guarded by _LOCK) ────────────────────────────────────────
_LOCK = threading.RLock()
_RUNTIME_CACHE: "dict[str, FoundryRuntime]" = {}
_EPS_REGISTERED = False


class FoundryLocalError(RuntimeError):
    """Raised when Foundry Local provisioning or discovery fails.

    Carries an actionable, user-facing message (missing SDK, service won't
    start, model not cached, etc.).
    """


@dataclass(frozen=True)
class FoundryModel:
    """A model entry in the Foundry Local catalog.

    ``alias`` is the stable, portable, human-friendly name (e.g.
    ``"qwen2.5-0.5b"``) — this is what Hermes persists in config.yaml. ``id`` is
    the concrete hardware-specific variant id the local server expects in the
    OpenAI ``model`` field (e.g. ``"qwen2.5-0.5b-instruct-generic-cpu"``).
    """

    alias: str
    id: str
    cached: bool = False
    loaded: bool = False


@dataclass(frozen=True)
class FoundryRuntime:
    """A provisioned, ready-to-use Foundry Local endpoint.

    ``base_url`` ends in ``/v1`` (OpenAI-compatible). ``model_id`` is the
    concrete variant id to send in the request ``model`` field — the runtime
    resolver surfaces this so the wire request uses the resolved id rather than
    the portable alias the user configured.
    """

    base_url: str
    api_key: str
    model_id: str
    alias: str


# ---------------------------------------------------------------------------
# SDK import / availability
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if a Foundry Local SDK can be imported right now.

    Cheap check — does not initialize the manager or start the service. Accepts
    either the current (``foundry_local_sdk``) or legacy (``foundry_local``)
    package so ``hermes doctor`` / setup preflight reflect reality.
    """
    try:
        import foundry_local_sdk  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import foundry_local  # noqa: F401
        return True
    except Exception:
        return False


def _import_sdk(*, allow_install: bool = True) -> Tuple[Any, str]:
    """Import the Foundry Local SDK, returning ``(module, kind)``.

    ``kind`` is ``"current"`` for the 1.x ``foundry_local_sdk`` package or
    ``"legacy"`` for the 0.x ``foundry_local`` package. Prefers the current
    package when both are importable.

    Lazy-installs ``foundry-local-sdk`` (subject to ``security.allow_lazy_installs``)
    when nothing is importable and ``allow_install`` is True.

    Raises :class:`FoundryLocalError` with an actionable message when the SDK
    cannot be imported or installed.
    """
    try:
        import foundry_local_sdk as _mod
        return _mod, "current"
    except ImportError:
        pass
    try:
        import foundry_local as _legacy
        return _legacy, "legacy"
    except ImportError:
        pass

    if not allow_install:
        raise FoundryLocalError(_install_hint())

    # Try the standard Hermes lazy-install path (gated by config).
    try:
        from tools.lazy_deps import ensure, FeatureUnavailable
    except ImportError as exc:  # pragma: no cover — tools always present
        raise FoundryLocalError(_install_hint()) from exc
    try:
        ensure(_FOUNDRY_LOCAL_FEATURE, prompt=False)
    except FeatureUnavailable as exc:
        raise FoundryLocalError(_install_hint(extra=str(exc))) from exc
    except Exception as exc:  # pragma: no cover — defensive
        raise FoundryLocalError(_install_hint(extra=str(exc))) from exc

    # Retry after install.
    try:
        import foundry_local_sdk as _mod  # noqa: WPS440
        return _mod, "current"
    except ImportError:
        try:
            import foundry_local as _legacy  # noqa: WPS440
            return _legacy, "legacy"
        except ImportError as exc:
            raise FoundryLocalError(_install_hint()) from exc


def _install_hint(*, extra: str = "") -> str:
    base = (
        "Foundry Local requires the 'foundry-local-sdk' package and the "
        "Foundry Local runtime. Install the SDK with:\n"
        "    pip install foundry-local-sdk openai        (cross-platform)\n"
        "    pip install foundry-local-sdk-winml openai  (Windows, hardware-accelerated)\n"
        "and install the runtime from "
        "https://learn.microsoft.com/azure/foundry-local/get-started"
    )
    if extra:
        base += f"\n({extra})"
    return base


# ---------------------------------------------------------------------------
# Manager bootstrap (current 1.x SDK — singleton + web service)
# ---------------------------------------------------------------------------


def _build_configuration(mod: Any, *, fixed_url: str = "") -> Any:
    """Build a ``Configuration`` for the current SDK.

    ``fixed_url`` pins the web service to a specific ``host:port`` (without the
    ``/v1`` suffix) when the user has set ``FOUNDRY_LOCAL_BASE_URL`` — useful for
    a stable endpoint that external tools can also reach. When empty, the SDK
    picks a free port and we read it back from ``manager.urls``.
    """
    kwargs: dict = {"app_name": _APP_NAME}
    if fixed_url:
        kwargs["web"] = {"urls": fixed_url}
    return mod.Configuration(**kwargs)


def _current_manager(mod: Any, *, fixed_url: str = "") -> Any:
    """Return the initialized singleton ``FoundryLocalManager`` (1.x SDK)."""
    manager_cls = mod.FoundryLocalManager
    instance = getattr(manager_cls, "instance", None)
    if instance is None:
        config = _build_configuration(mod, fixed_url=fixed_url)
        manager_cls.initialize(config)
        instance = manager_cls.instance
    return instance


def _register_eps(manager: Any, on_progress: Optional[ProgressCallback]) -> None:
    """Download + register execution providers (Windows ML / ONNX Runtime EPs).

    One-time, potentially slow (downloads CUDA/QNN/OpenVINO packages). Cached
    per-process via ``_EPS_REGISTERED`` so it runs at most once. Best-effort:
    on cross-platform builds there is nothing to download and the SDK returns
    immediately; failures here are surfaced because ``model.load()`` depends on
    EPs being present.
    """
    global _EPS_REGISTERED
    if _EPS_REGISTERED:
        return
    register = getattr(manager, "download_and_register_eps", None)
    if register is None:
        _EPS_REGISTERED = True
        return

    def _cb(ep_name: str, percent: float) -> None:
        if on_progress is not None:
            try:
                on_progress(f"ep:{ep_name}", float(percent))
            except Exception:
                pass

    try:
        register(progress_callback=_cb)
    except TypeError:
        # Older signature without the keyword.
        register(_cb)
    _EPS_REGISTERED = True


def _strip_v1(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")]
    return u


def _endpoint_from_manager(manager: Any) -> str:
    """Read the OpenAI-compatible base URL (ending in ``/v1``) from a 1.x manager."""
    urls = getattr(manager, "urls", None)
    if not urls:
        raise FoundryLocalError(
            "Foundry Local web service did not report an endpoint. The service "
            "may have failed to start — check the Foundry Local installation."
        )
    base = _strip_v1(urls[0])
    if not base:
        raise FoundryLocalError("Foundry Local returned an empty endpoint URL.")
    return f"{base}/v1"


# ---------------------------------------------------------------------------
# Public: catalog listing
# ---------------------------------------------------------------------------


def list_models(*, allow_install: bool = False, timeout: float = 30.0) -> List[FoundryModel]:
    """List models available in the Foundry Local catalog for this device.

    Used by the interactive setup flow to populate the model picker. Returns an
    empty list if the catalog can't be read. Does not download anything.

    ``allow_install`` defaults False so a hot path (e.g. ``hermes doctor``) never
    triggers a pip install; the setup flow passes True.
    """
    mod, kind = _import_sdk(allow_install=allow_install)
    with _LOCK:
        if kind == "current":
            return _list_models_current(mod)
        return _list_models_legacy(mod)


def _model_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def _to_model(raw: Any) -> Optional[FoundryModel]:
    alias = _model_attr(raw, "alias", "name", default="")
    mid = _model_attr(raw, "id", "model_id", default="")
    if not (alias or mid):
        return None
    cached = bool(_model_attr(raw, "is_cached", "cached", default=False))
    loaded = bool(_model_attr(raw, "is_loaded", "loaded", default=False))
    return FoundryModel(
        alias=str(alias or mid),
        id=str(mid or alias),
        cached=cached,
        loaded=loaded,
    )


def _list_models_current(mod: Any) -> List[FoundryModel]:
    manager = _current_manager(mod)
    catalog = getattr(manager, "catalog", None)
    raw_models: Any = []
    if catalog is not None:
        lister = getattr(catalog, "list_models", None) or getattr(catalog, "get_models", None)
        if lister is not None:
            raw_models = lister() or []
    out: List[FoundryModel] = []
    for raw in raw_models:
        m = _to_model(raw)
        if m is not None:
            out.append(m)
    return out


def _list_models_legacy(mod: Any) -> List[FoundryModel]:
    # Legacy 0.x: FoundryLocalManager() with list_catalog_models().
    manager = mod.FoundryLocalManager()
    lister = getattr(manager, "list_catalog_models", None)
    raw_models = lister() if lister is not None else []
    out: List[FoundryModel] = []
    for raw in raw_models or []:
        m = _to_model(raw)
        if m is not None:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Public: provision (download/load) + connect (endpoint discovery)
# ---------------------------------------------------------------------------


def provision(
    model_ref: str,
    *,
    allow_download: bool = False,
    allow_install: bool = False,
    fixed_url: str = "",
    on_progress: Optional[ProgressCallback] = None,
    force: bool = False,
) -> FoundryRuntime:
    """Ensure ``model_ref`` is loaded and return its OpenAI-compatible endpoint.

    ``model_ref`` may be a portable alias (``"qwen2.5-0.5b"``) or a concrete
    variant id. On the first call this starts the Foundry Local service and
    loads the model (downloading it first when ``allow_download`` is True); the
    result is cached per-process so subsequent per-turn calls are instant.

    Parameters
    ----------
    allow_download:
        When True (interactive setup), download the model if it isn't cached and
        register execution providers. When False (per-turn runtime resolution),
        never trigger a multi-GB download mid-session — raise a clear error
        telling the user to run ``hermes model`` instead.
    allow_install:
        When True, lazy-install the SDK if missing. Defaults False for hot
        runtime paths.
    fixed_url:
        Pin the web service to this ``host:port`` (``/v1`` optional). Usually
        sourced from ``FOUNDRY_LOCAL_BASE_URL``.
    on_progress:
        Optional ``(stage, percent)`` callback for download / EP progress.
    force:
        Bypass the per-process cache and re-provision.

    Raises :class:`FoundryLocalError` on any failure.
    """
    ref = (model_ref or "").strip()
    if not ref:
        raise FoundryLocalError(
            "No Foundry Local model configured. Run 'hermes model' and pick a "
            "Foundry Local model."
        )

    cache_key = f"{ref}@{_strip_v1(fixed_url)}"
    if not force:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached

    mod, kind = _import_sdk(allow_install=allow_install)
    with _LOCK:
        if not force:
            cached = _RUNTIME_CACHE.get(cache_key)
            if cached is not None:
                return cached
        try:
            if kind == "current":
                runtime = _provision_current(
                    mod,
                    ref,
                    allow_download=allow_download,
                    fixed_url=_strip_v1(fixed_url),
                    on_progress=on_progress,
                )
            else:
                runtime = _provision_legacy(mod, ref)
        except FoundryLocalError:
            raise
        except Exception as exc:  # normalize SDK errors into a clear message
            raise FoundryLocalError(
                f"Foundry Local failed to provision '{ref}': {exc}"
            ) from exc
        _RUNTIME_CACHE[cache_key] = runtime
        return runtime


def _provision_current(
    mod: Any,
    ref: str,
    *,
    allow_download: bool,
    fixed_url: str,
    on_progress: Optional[ProgressCallback],
) -> FoundryRuntime:
    manager = _current_manager(mod, fixed_url=fixed_url)

    catalog = getattr(manager, "catalog", None)
    if catalog is None:
        raise FoundryLocalError("Foundry Local SDK exposes no model catalog.")

    get_model = getattr(catalog, "get_model", None)
    if get_model is None:
        raise FoundryLocalError("Foundry Local catalog has no get_model().")
    model = get_model(ref)
    if model is None:
        raise FoundryLocalError(
            f"Foundry Local model '{ref}' not found in the catalog for this "
            "device. Run 'hermes model' to pick an available model."
        )

    is_cached = bool(_model_attr(model, "is_cached", default=False))
    if not is_cached:
        if not allow_download:
            raise FoundryLocalError(
                f"Foundry Local model '{ref}' is not downloaded yet. Run "
                "'hermes model' (and pick it) to download it once, then retry."
            )
        _register_eps(manager, on_progress)
        _download_model(model, on_progress)

    is_loaded = bool(_model_attr(model, "is_loaded", default=False))
    if not is_loaded:
        load = getattr(model, "load", None)
        if load is None:
            raise FoundryLocalError("Foundry Local model object has no load().")
        load()

    # Expose the OpenAI-compatible REST endpoint.
    start_web = getattr(manager, "start_web_service", None)
    if start_web is not None:
        start_web()

    base_url = _endpoint_from_manager(manager)
    model_id = str(_model_attr(model, "id", "model_id", default=ref) or ref)
    alias = str(_model_attr(model, "alias", default=ref) or ref)
    return FoundryRuntime(
        base_url=base_url,
        api_key=LOCAL_PLACEHOLDER_API_KEY,
        model_id=model_id,
        alias=alias,
    )


def _download_model(model: Any, on_progress: Optional[ProgressCallback]) -> None:
    download = getattr(model, "download", None)
    if download is None:
        return

    def _cb(percent: float) -> None:
        if on_progress is not None:
            try:
                on_progress("download", float(percent))
            except Exception:
                pass

    try:
        download(_cb)
    except TypeError:
        # Some versions take no progress callback.
        download()


def _provision_legacy(mod: Any, ref: str) -> FoundryRuntime:
    # Legacy 0.x: the constructor bootstraps (start service + download + load).
    manager = mod.FoundryLocalManager(ref)
    endpoint = str(getattr(manager, "endpoint", "") or "").strip()
    if not endpoint:
        raise FoundryLocalError(
            "Legacy Foundry Local SDK did not expose an endpoint. Upgrade with: "
            "pip install -U foundry-local-sdk"
        )
    api_key = str(getattr(manager, "api_key", "") or "") or LOCAL_PLACEHOLDER_API_KEY
    model_id = ref
    get_info = getattr(manager, "get_model_info", None)
    if get_info is not None:
        try:
            info = get_info(ref)
            model_id = str(_model_attr(info, "id", "model_id", default=ref) or ref)
        except Exception:
            model_id = ref
    base_url = endpoint if endpoint.rstrip("/").endswith("/v1") else f"{endpoint.rstrip('/')}/v1"
    return FoundryRuntime(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        alias=ref,
    )


# ---------------------------------------------------------------------------
# Lifecycle / test helpers
# ---------------------------------------------------------------------------


def reset_state() -> None:
    """Clear the per-process provisioning cache. Used by tests + profile switches."""
    global _EPS_REGISTERED
    with _LOCK:
        _RUNTIME_CACHE.clear()
        _EPS_REGISTERED = False


__all__ = [
    "FoundryLocalError",
    "FoundryModel",
    "FoundryRuntime",
    "LOCAL_PLACEHOLDER_API_KEY",
    "ProgressCallback",
    "is_available",
    "list_models",
    "provision",
    "reset_state",
]
