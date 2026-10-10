# langchain-tanvo

[![PyPI](https://img.shields.io/pypi/v/langchain-tanvo)](https://pypi.org/project/langchain-tanvo/) [![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

LangChain tools for [Tanvo](https://tanvo.ai/?utm_source=github&utm_medium=referral), an AI image, video and music studio. Give an agent the ability to make images (Nano Banana, Seedream, GPT Image, Qwen, Grok Imagine), short videos with sound (Kling, Veo, Seedance, Wan, MiniMax) and songs (Suno V6), or to run one of 80 ready-made photo apps (pet portraits, figurines, old photo restoration, trend videos) on the user's photos.

- Works with LangChain 1.x agents (`create_agent`) and LangGraph.
- **No API key needed to try it**: the house image engine runs on Tanvo's free tier.
- Each tool returns a short text for the model (status, cost, file URLs) and the full run as an artifact for your code.
- When a setting is rejected, the error lists what the model does accept, so the agent can correct itself.

```bash
pip install langchain-tanvo
```

## Tools

| Tool | Name the model sees | What it does |
|---|---|---|
| `TanvoFindApps` | `tanvo_find_apps` | Find a ready-made app for what the user wants |
| `TanvoRunApp` | `tanvo_run_app` | Run an app's preset look on the user's photos |
| `TanvoImageGeneration` | `tanvo_image_generation` | Text to image, or edit / combine photos |
| `TanvoVideoGeneration` | `tanvo_video_generation` | Text or image to video, optional end frame and sound |
| `TanvoMusicGeneration` | `tanvo_music_generation` | A song: describe it, bring lyrics, or go instrumental |
| `TanvoGetGeneration` | `tanvo_get_generation` | Status and files of an earlier run |

`TanvoToolkit().get_tools()` returns all six, sharing one client.

## Setup

```bash
export TANVO_API_KEY="sk_..."   # optional; create one at https://tanvo.ai/settings/apikeys
```

Without a key the tools use Tanvo's anonymous free tier, which runs the house image engine (`studio-image-v1`) only. A key runs on your account: every model, the same prices as the website, and every run in your history. Failed runs are refunded automatically.

## Use with an agent

```python
from langchain.agents import create_agent
from langchain_tanvo import TanvoToolkit

agent = create_agent("anthropic:claude-sonnet-5-5", tools=TanvoToolkit().get_tools())
result = agent.invoke({"messages": [{"role": "user", "content": "Make my dog a royal portrait: https://example.com/rex.jpg"}]})
print(result["messages"][-1].content)
```

The agent finds the pet portrait app with `tanvo_find_apps`, runs its `royal` look with `tanvo_run_app` and answers with the image URL.

## Use a tool directly

```python
from langchain_tanvo import TanvoImageGeneration

tool = TanvoImageGeneration()
print(tool.invoke({"prompt": "a tiny lighthouse on a rocky island at dusk, film photo", "aspect_ratio": "4:3"}))

# As a tool call, to also get the artifact (the full run)
msg = tool.invoke({"type": "tool_call", "name": tool.name, "id": "1", "args": {"prompt": "…", "model": "nano-banana-2", "resolution": "2K"}})
print(msg.content)            # "Done (image, nano-banana-2, 25 credits). Files: - https://media.tanvo.ai/…"
print(msg.artifact["outputs"])
```

Video waits for the clip by default (usually 1–8 minutes). For long-running agents, return at once and check later:

```python
from langchain_tanvo import TanvoGetGeneration, TanvoVideoGeneration

tools = [TanvoVideoGeneration(wait=False), TanvoGetGeneration()]   # or TanvoToolkit(video_wait=False)
```

## Configuration

| Parameter | Tools | |
|---|---|---|
| `api_key` | all | Defaults to `TANVO_API_KEY` |
| `base_url` | all | Defaults to `TANVO_BASE_URL`, then `https://tanvo.ai` |
| `client` | all | A `tanvo.Client` to share between tools |
| `default_model` | image, video | Used when the call names no model (`studio-image-v1`, `studio-video-v1`) |
| `wait`, `timeout` | video | Wait for the clip (default `True`, 600 s) |
| `limit` | find apps | How many apps to return (default 6) |

Models, their options and prices: [tanvo.ai/developers/api](https://tanvo.ai/developers/api?utm_source=github&utm_medium=referral#models).

## Also from Tanvo

- [tanvo-python](https://github.com/tanvoai/tanvo-python): the Python client these tools are built on
- [tanvo-mcp](https://github.com/tanvoai/tanvo-mcp): the same apps and models in Claude, Cursor and other MCP clients
- [tanvo-ai-provider](https://github.com/tanvoai/tanvo-ai-provider): a provider for the Vercel AI SDK

## License

MIT
