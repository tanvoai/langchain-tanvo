"""Unit tests: LangChain's standard tool tests, plus each tool driven against a fake tanvo.Client (no network)."""

from __future__ import annotations

from typing import Any

import pytest
import tanvo
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_tanvo import (
    TanvoFindApps,
    TanvoGetGeneration,
    TanvoImageGeneration,
    TanvoMusicGeneration,
    TanvoRunApp,
    TanvoToolkit,
    TanvoVideoGeneration,
)


def gen(kind="image", status="SUCCEEDED", **kw) -> tanvo.Generation:
    outputs = [tanvo.Output(url=f"https://media.tanvo.ai/outputs/g1/0.{'mp4' if kind == 'video' else 'mp3' if kind == 'music' else 'png'}", mime="x", title="Ana Rides" if kind == "music" else None)] if status == "SUCCEEDED" else []
    return tanvo.Generation(id="g1", kind=kind, model=kw.get("model", "m"), prompt="p", options={}, cost=25, status=status, error=kw.get("error"), created_at=1, outputs=outputs)


APPS = [
    tanvo.App.from_dict({
        "slug": "ai-pet-portrait-generator", "kind": "image", "name": "Pet Portrait Generator", "description": "A painted portrait of the one who runs the house.",
        "category": "Pets", "model": "seedream-5-lite", "url": "https://tanvo.ai/image/ai-pet-portrait-generator",
        "inputs": [{"id": "in1", "label": "Pet photo", "hint": "", "min": 1, "max": 1}],
        "looks": [{"id": "renaissance", "name": "Renaissance oil", "prompt": "oil portrait", "model": "seedream-5-lite", "options": {}, "url": ""}, {"id": "royal", "name": "Royal monarch", "prompt": "monarch", "model": "seedream-5-lite", "options": {}, "url": ""}],
    }),
    tanvo.App.from_dict({"slug": "old-photo-restoration", "kind": "image", "name": "Old Photo Restoration", "description": "Repair scratches and fading.", "category": "Photo fixes", "model": "m", "url": "", "inputs": [], "looks": [{"id": "default", "name": "Restore", "prompt": "restore", "model": "m", "options": {}, "url": ""}]}),
]


class FakeClient:
    def __init__(self, result: tanvo.Generation | None = None, error: Exception | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.result, self.error = result, error

    def _call(self, name: str, *a: Any, **kw: Any):
        self.calls.append((name, a, kw))
        if self.error:
            raise self.error
        return self.result

    def generate_image(self, *a, **kw):
        return self._call("generate_image", *a, **kw)

    def generate_video(self, *a, **kw):
        return self._call("generate_video", *a, **kw)

    def generate_music(self, *a, **kw):
        return self._call("generate_music", *a, **kw)

    def generate_from_app(self, *a, **kw):
        return self._call("generate_from_app", *a, **kw)

    def get(self, *a, **kw):
        return self._call("get", *a, **kw)

    def model(self, model_id):
        if model_id != "studio-image-v1":
            raise KeyError(model_id)
        return tanvo.Model.from_dict({"id": model_id, "options": {"aspect": ["1:1", "16:9", "4:3"], "resolution": ["1K"], "tier": None}})

    def apps(self, kind=None):
        self.calls.append(("apps", (kind,), {}))
        return [x for x in APPS if kind is None or x.kind == kind]


def call(tool: BaseTool, args: dict) -> ToolMessage:
    return tool.invoke({"type": "tool_call", "name": tool.name, "id": "call_1", "args": args})


# ─── LangChain standard tests ────────────────────────────────────────────────


class _Standard(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict:
        return {"api_key": "sk_test"}

    @property
    def init_from_env_params(self):
        return {"TANVO_API_KEY": "sk_from_env"}, {}, {}


class TestImageStandard(_Standard):
    tool_constructor = TanvoImageGeneration
    tool_invoke_params_example = {"prompt": "a lighthouse in fog", "aspect_ratio": "16:9"}


class TestVideoStandard(_Standard):
    tool_constructor = TanvoVideoGeneration
    tool_invoke_params_example = {"prompt": "a fox running in snow", "duration": 5}


class TestMusicStandard(_Standard):
    tool_constructor = TanvoMusicGeneration
    tool_invoke_params_example = {"prompt": "a birthday song for Sam", "mode": "Describe"}


class TestFindAppsStandard(_Standard):
    tool_constructor = TanvoFindApps
    tool_invoke_params_example = {"query": "royal pet portrait"}


class TestRunAppStandard(_Standard):
    tool_constructor = TanvoRunApp
    tool_invoke_params_example = {"app": "ai-pet-portrait-generator", "look": "royal", "photos": ["https://example.com/rex.jpg"]}


class TestGetGenerationStandard(_Standard):
    tool_constructor = TanvoGetGeneration
    tool_invoke_params_example = {"id": "g1"}


# ─── Behaviour ───────────────────────────────────────────────────────────────


def test_api_key_from_env_and_free_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TANVO_API_KEY", "sk_env")
    assert TanvoImageGeneration().api_key.get_secret_value() == "sk_env"
    monkeypatch.delenv("TANVO_API_KEY")
    t = TanvoImageGeneration()
    assert t.api_key is None
    assert t._tanvo().anon_id.startswith("py-")  # the free tier


def test_image_passes_arguments_and_returns_url_and_artifact() -> None:
    fake = FakeClient(gen("image", model="nano-banana-2"))
    msg = call(TanvoImageGeneration(client=fake), {"prompt": "a lighthouse", "model": "nano-banana-2", "aspect_ratio": "16:9", "resolution": "2K", "images": ["https://x/a.jpg"]})
    name, args, kw = fake.calls[0]
    assert (name, args[0]) == ("generate_image", "a lighthouse")
    assert kw == {"model": "nano-banana-2", "aspect": "16:9", "resolution": "2K", "images": ["https://x/a.jpg"]}
    assert "https://media.tanvo.ai/outputs/g1/0.png" in msg.content and "25 credits" in msg.content
    assert msg.artifact["id"] == "g1" and msg.artifact["outputs"][0]["url"].endswith(".png")


def test_image_uses_the_default_model() -> None:
    fake = FakeClient(gen())
    call(TanvoImageGeneration(client=fake, default_model="seedream-5-lite"), {"prompt": "x"})
    assert fake.calls[0][2]["model"] == "seedream-5-lite"


def test_video_waits_by_default_and_can_return_a_run_id() -> None:
    fake = FakeClient(gen("video"))
    call(TanvoVideoGeneration(client=fake), {"prompt": "a fox", "model": "kling-3-0", "duration": 5, "image": "https://x/a.jpg", "audio": True})
    kw = fake.calls[0][2]
    assert kw["wait"] is True and kw["model"] == "kling-3-0" and kw["duration"] == 5 and kw["image"] == "https://x/a.jpg" and kw["audio"] is True
    fake = FakeClient(gen("video", status="PROCESSING"))
    msg = call(TanvoVideoGeneration(client=fake, wait=False), {"prompt": "a fox"})
    assert fake.calls[0][2]["wait"] is False
    assert "tanvo_get_generation" in msg.content and "g1" in msg.content


def test_music() -> None:
    fake = FakeClient(gen("music"))
    msg = call(TanvoMusicGeneration(client=fake), {"prompt": "[Verse] …", "mode": "Lyrics", "title": "Ana Rides"})
    assert fake.calls[0][2] == {"mode": "Lyrics", "style": None, "title": "Ana Rides"}
    assert "Ana Rides: https://media.tanvo.ai/outputs/g1/0.mp3" in msg.content


def test_find_apps_ranks_by_intent() -> None:
    fake = FakeClient()
    msg = call(TanvoFindApps(client=fake), {"query": "make my dog look royal"})
    assert msg.content.splitlines()[1].startswith("- Pet Portrait Generator (app='ai-pet-portrait-generator'")
    assert "royal" in msg.content
    assert msg.artifact["apps"][0]["slug"] == "ai-pet-portrait-generator"
    msg = call(TanvoFindApps(client=fake), {"query": "zzzz nothing"})
    assert "No Tanvo app matched" in msg.content


def test_run_app() -> None:
    fake = FakeClient(gen())
    call(TanvoRunApp(client=fake), {"app": "ai-pet-portrait-generator", "look": "royal", "photos": ["https://x/rex.jpg"], "details": "Rex"})
    assert fake.calls[0] == ("generate_from_app", ("ai-pet-portrait-generator",), {"look": "royal", "photos": ["https://x/rex.jpg"], "details": "Rex"})


def test_errors_become_tool_messages_not_crashes() -> None:
    fake = FakeClient(error=tanvo.QuotaError("needs_api_key", "The free tier covers the house image engine only. Get a key at https://tanvo.ai/settings/apikeys", 403))
    msg = call(TanvoVideoGeneration(client=fake), {"prompt": "x"})
    assert msg.status == "error" and "needs_api_key" in msg.content and "settings/apikeys" in msg.content
    msg = call(TanvoGetGeneration(client=FakeClient(gen(status="FAILED", error="upstream timeout"))), {"id": "g1"})
    assert msg.status == "error" and "refunded" in msg.content
    msg = call(TanvoRunApp(client=FakeClient(error=KeyError("Unknown app 'x'"))), {"app": "x"})
    assert msg.status == "error"


def test_rejected_settings_list_what_the_model_accepts() -> None:
    fake = FakeClient(error=tanvo.InvalidRequest("invalid", "That model or setting is not available.", 400))
    msg = call(TanvoImageGeneration(client=fake), {"prompt": "x", "aspect_ratio": "3:2"})
    assert msg.status == "error"
    assert "Settings studio-image-v1 accepts: aspect: 1:1, 16:9, 4:3; resolution: 1K." in msg.content
    msg = call(TanvoImageGeneration(client=fake), {"prompt": "x", "model": "no-such-model"})
    assert "Unknown model 'no-such-model'" in msg.content


def test_toolkit_shares_one_client() -> None:
    tools = TanvoToolkit(api_key="sk_test", video_wait=False).get_tools()
    assert [t.name for t in tools] == ["tanvo_find_apps", "tanvo_run_app", "tanvo_image_generation", "tanvo_video_generation", "tanvo_music_generation", "tanvo_get_generation"]
    assert len({id(t.client) for t in tools}) == 1
    assert next(t for t in tools if t.name == "tanvo_video_generation").wait is False
