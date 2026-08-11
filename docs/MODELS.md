# Frozen contemporary-model registry

The prospective protocol freezes three model families on 2026-08-11.

## DeepSeek-V4-Flash
- Registry ID: `deepseek-v4-flash`
- Hosted API base: `https://api.deepseek.com`
- Secret: `DEEPSEEK_API_KEY`
- Thinking mode is explicitly disabled in the frozen request overrides so the declared temperature parameter is operative and the task remains a short A/B/C response.

## Mistral Small 4 v26.03
- Registry ID: `mistral-small-2603`
- Hosted API base: `https://api.mistral.ai/v1`
- Secret: `MISTRAL_API_KEY`

## Qwen3.5-4B
- Frozen model ID: `Qwen/Qwen3.5-4B`
- Open-weight model served through an OpenAI-compatible endpoint.
- Local default: `http://127.0.0.1:8000/v1`
- GitHub Actions uses repository variable `QWEN_BASE_URL` and secret `QWEN_API_KEY` to reach a hosted endpoint serving this exact frozen model.

Changing a model ID changes the prospective protocol. If a researcher intentionally substitutes another model, the resulting run should be labeled as a new model condition rather than silently treated as the frozen condition.
