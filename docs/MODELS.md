# Frozen contemporary-model registry

The prospective validation protocol freezes the following model identifiers as of **2026-08-11**. These are execution targets, not reported empirical results in the accompanying manuscript.

| Registry name | Model identifier | Execution mode | Official source |
|---|---|---|---|
| DeepSeek-V4-Flash | `deepseek-v4-flash` | DeepSeek OpenAI-compatible API | https://api-docs.deepseek.com/quick_start/pricing |
| Mistral Small 4 v26.03 | `mistral-small-2603` | Mistral Chat Completions API | https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Local OpenAI-compatible server (e.g. vLLM) | https://huggingface.co/Qwen/Qwen3.5-4B |

API providers may change availability, aliases, pricing, rate limits, or behavior after the frozen date. Reproduction should record the actual execution date, endpoint, model identifier returned by the provider where available, and any provider-side revision metadata.

No API credentials are included in the repository.
