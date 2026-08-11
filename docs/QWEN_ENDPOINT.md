# Qwen3.5-4B endpoint options

The frozen prospective condition is `Qwen/Qwen3.5-4B`.

## Local execution

On hardware capable of serving the model, expose it through an OpenAI-compatible server and keep the served model ID equal to `Qwen/Qwen3.5-4B`. One common approach is an OpenAI-compatible vLLM server. The E-EA repository intentionally does not install GPU-serving packages in the base `requirements.txt` because they are hardware-specific.

The default local URL in `model_registry.json` is:

`http://127.0.0.1:8000/v1`

For a local endpoint that does not require authentication, `QWEN_API_KEY` may be left unset.

## GitHub Actions

Standard GitHub-hosted runners orchestrate API calls but are not intended to host this model locally. For the GitHub workflow, deploy the exact frozen model on a compatible hosted OpenAI-style inference endpoint, then set:

- repository secret `QWEN_API_KEY`
- repository variable `QWEN_BASE_URL`

Do not silently substitute another Qwen model ID. A substituted model is a new experimental condition and should be reported as such.
