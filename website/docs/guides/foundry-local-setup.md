---
sidebar_position: 10
title: "Run Hermes On-Device with Foundry Local"
description: "Run Hermes Agent entirely on your own machine with Microsoft Foundry Local — hardware-optimized on-device models, no API keys, no cloud."
---

# Run Hermes On-Device with Foundry Local

[Foundry Local](https://learn.microsoft.com/azure/foundry-local/) is Microsoft's
on-device inference runtime. It downloads hardware-optimized ONNX model variants
(CPU / CUDA / NPU / Vitis / QNN / OpenVINO) and serves them over a local
OpenAI-compatible endpoint. Hermes drives it through the same chat-completions
path it uses for LM Studio and Ollama, so once configured the agent works exactly
as it does with any cloud provider — terminal commands, file editing, web
browsing, delegation — but the model runs locally with no API key and no data
leaving your machine.

:::note Foundry Local vs. Azure Foundry
This is the **on-device** runtime (provider `foundry-local`). It is distinct from
the cloud **Azure AI Foundry** service (provider `azure-foundry`), which targets a
remote Azure endpoint with an API key or Microsoft Entra ID.
:::

## What You Need

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/11, macOS (Apple Silicon), or Linux | Windows 11 with an NPU/GPU |
| **Python** | 3.11+ | 3.11+ |
| **RAM** | 8 GB (for ~0.5–3B models) | 16+ GB |
| **GPU/NPU** | Not required (CPU works) | CUDA GPU / NPU for speed |

## Step 1: Install the runtime and SDK

Install the Foundry Local runtime from Microsoft's
[get-started guide](https://learn.microsoft.com/azure/foundry-local/get-started),
then install the Python SDK Hermes uses to provision and connect:

```bash
pip install foundry-local-sdk openai          # cross-platform
# or, on Windows for hardware acceleration:
pip install foundry-local-sdk-winml openai
```

Hermes can also lazy-install the SDK for you the first time you select the
provider (subject to `security.allow_lazy_installs`).

## Step 2: Configure Hermes

Run the interactive picker:

```bash
hermes model
```

Choose **Foundry Local**. Hermes reads the on-device catalog, lets you pick a
model alias (e.g. `qwen2.5-0.5b`), then downloads and loads it once. The chosen
**alias** is saved to `config.yaml` — it's portable across machines because the
SDK resolves the right hardware variant for each device at runtime.

```yaml
# ~/.hermes/config.yaml
model:
  provider: "foundry-local"
  default: "qwen2.5-0.5b"   # portable alias; resolved to the hardware variant at runtime
```

There is **no base URL or API key** to set — the SDK provisions a local
OpenAI-compatible endpoint dynamically and Hermes discovers it on each run.

## Step 3: Chat

```bash
hermes chat
```

On the first message Hermes starts the Foundry Local web service (if it isn't
already running), ensures the model is loaded, and connects. Subsequent turns
reuse the cached endpoint.

## Optional: pin a fixed port

By default the SDK binds the local web service to a free port. To pin it (useful
if other tools also talk to the endpoint), set:

```bash
# ~/.hermes/.env
FOUNDRY_LOCAL_BASE_URL=http://127.0.0.1:5273
```

## Tips for slow hardware

Small local models can be slow on CPU. Widen the API timeout (an env var, not a
`config.yaml` key):

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30 minutes — generous for slow local models
```

Foundry Local fits Hermes' weak-hardware focus well: pick a small alias
(`qwen2.5-0.5b`, `phi-3.5-mini`), keep one model loaded, and let the SDK select
the best execution provider your device supports.

## Troubleshooting

- **`foundry-local-sdk` not installed** — `pip install foundry-local-sdk openai`,
  or enable lazy installs.
- **"model is not downloaded yet"** at chat time — run `hermes model` and pick the
  model once so it downloads; runtime never triggers a multi-GB download
  mid-session.
- **No models listed** — confirm the runtime is installed and `foundry model list`
  works in your shell.
- **Model not found** — run `hermes model` again to see the aliases available for
  your device's hardware.
