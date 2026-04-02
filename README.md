# pi-extension-claude-vertex

A [pi](https://github.com/mariozechner/pi) extension that adds Claude models via **Google Cloud Vertex AI**, using Application Default Credentials (ADC) — no API key needed.

## Prerequisites

- A GCP project with the Vertex AI API enabled
- `gcloud` CLI authenticated: `gcloud auth application-default login`
- The following environment variables set:

| Variable | Description |
|---|---|
| `ANTHROPIC_VERTEX_PROJECT_ID` | Your GCP project ID |
| `CLOUD_ML_REGION` | Vertex AI region, e.g. `us-east5` or `global` |

> **Already using Claude Code with Vertex?** If you have `CLAUDE_CODE_USE_VERTEX=1` in your environment, these variables are already set. Just install and go.

## Install

```bash
cd ~/.pi/agent/extensions/claude-vertex
npm install
```

Then run pi normally and select a model under the **claude-vertex** provider via `/model`.

## Models

| Model | ID |
|---|---|
| Claude Opus 4.6 | `claude-opus-4-6@default` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6@default` |
| Claude Haiku 4.5 | `claude-haiku-4-5@20251001` |

All models support extended thinking / reasoning.

## How it works

The extension registers a custom `claude-vertex` provider that calls the Vertex AI API directly via the [`@anthropic-ai/vertex-sdk`](https://github.com/anthropics/anthropic-sdk-python/tree/main/anthropic-vertex) package. Authentication is handled entirely by GCP Application Default Credentials — the same credentials used by `gcloud` and Claude Code.

Features:
- Streaming responses
- Extended thinking / reasoning
- Tool use
- Vision (images)
- Prompt caching
