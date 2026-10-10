"""LangChain tools for Tanvo (https://tanvo.ai): images, video and songs on 30 models, plus ready-made photo apps.

Every tool returns `(content, artifact)`: `content` is a short text for the model (status, cost, file URLs) and
`artifact` is the full run as a dict (`Generation` fields), for your own code.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Literal, Optional

import tanvo
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

__all__ = [
    "TanvoImageGeneration",
    "TanvoVideoGeneration",
    "TanvoMusicGeneration",
    "TanvoFindApps",
    "TanvoRunApp",
    "TanvoGetGeneration",
    "TanvoToolkit",
]


class _TanvoTool(BaseTool):
    """Shared settings: the API key (TANVO_API_KEY; without one, the anonymous free tier) and the client."""

    api_key: Optional[SecretStr] = Field(default_factory=secret_from_env("TANVO_API_KEY", default=None))
    """Tanvo API key. Defaults to TANVO_API_KEY; omit both to use the free tier (house image engine only)."""
    base_url: Optional[str] = None
    """Another deployment; defaults to TANVO_BASE_URL, then https://tanvo.ai."""
    client: Optional[Any] = Field(default=None, exclude=True)
    """A `tanvo.Client` to use instead of creating one (tests, shared clients)."""
    response_format: Literal["content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _client: Optional[tanvo.Client] = PrivateAttr(default=None)

    def _tanvo(self) -> tanvo.Client:
        if self.client is not None:
            return self.client
        if self._client is None:
            key = self.api_key.get_secret_value() if self.api_key else None
            self._client = tanvo.Client(key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _result(g: tanvo.Generation) -> tuple[str, dict[str, Any]]:
        artifact = asdict(g)
        if g.status == "SUCCEEDED":
            files = "\n".join(f"- {o.title + ': ' if o.title else ''}{o.url}" for o in g.outputs)
            return f"Done ({g.kind}, {g.model}, {g.cost} credits). Files:\n{files}", artifact
        if g.status == "FAILED":
            raise ToolException(f"The {g.kind} run {g.id} failed: {g.error or 'unknown error'}. Its credits were refunded.")
        return f"Still running ({g.kind}, {g.model}). Check it later with tanvo_get_generation and id {g.id}.", artifact

    def _error(self, e: tanvo.TanvoError, model: Optional[str] = None) -> ToolException:
        """The API's message, plus, for a rejected setting, what the model does accept so the agent can retry."""
        text = str(e)
        if isinstance(e, tanvo.InvalidRequest) and e.code == "invalid" and model:
            try:
                opts = self._tanvo().model(model).options
                allowed = "; ".join(f"{k}: {', '.join(map(str, v))}" for k, v in opts.items() if isinstance(v, list) and v)
                if allowed:
                    text += f" Settings {model} accepts: {allowed}."
            except (KeyError, tanvo.TanvoError):
                text += f" Unknown model {model!r}? List them at https://tanvo.ai/api/v1/models."
        return ToolException(text)


# ─── Images ──────────────────────────────────────────────────────────────────


class ImageInput(BaseModel):
    prompt: str = Field(description="What to make: subject, setting, light and style. For edits, what to change and what to keep.")
    model: Optional[str] = Field(default=None, description="Model id, e.g. nano-banana-2, seedream-5-lite, gpt-image-2-5, studio-image-v1 (free tier).")
    aspect_ratio: Optional[str] = Field(default=None, description="1:1, 16:9, 9:16, 4:3, 3:4 …")
    resolution: Optional[str] = Field(default=None, description="1K, 2K or 4K where the model supports it.")
    images: list[str] = Field(default_factory=list, description="Photos to edit or combine: https URLs or local file paths.")


class TanvoImageGeneration(_TanvoTool):
    """Generate or edit an image on Tanvo and return its URL.

    Setup:
        `pip install langchain-tanvo` and set `TANVO_API_KEY` (optional: without it only the free house engine runs).

    Example:
        ```python
        from langchain_tanvo import TanvoImageGeneration

        tool = TanvoImageGeneration()
        tool.invoke({"prompt": "a paper lantern over a misty lake at dawn", "aspect_ratio": "16:9"})
        ```
    """

    name: str = "tanvo_image_generation"
    description: str = (
        "Generate an image from a text prompt, or edit / combine photos when `images` are given, on Tanvo's image models "
        "(Nano Banana, Seedream, GPT Image, Qwen, Grok Imagine). Returns the image URL and the credits charged."
    )
    args_schema: type[BaseModel] = ImageInput
    default_model: str = "studio-image-v1"
    """Model used when the call does not name one."""

    def _run(self, prompt: str, model: Optional[str] = None, aspect_ratio: Optional[str] = None, resolution: Optional[str] = None, images: Optional[list[str]] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            g = self._tanvo().generate_image(prompt, model=model or self.default_model, aspect=aspect_ratio, resolution=resolution, images=images or [])
        except tanvo.TanvoError as e:
            raise self._error(e, model or self.default_model) from e
        return self._result(g)


# ─── Video ───────────────────────────────────────────────────────────────────


class VideoInput(BaseModel):
    prompt: str = Field(description="One scene: subject, action, camera movement, light and style.")
    model: Optional[str] = Field(default=None, description="Model id, e.g. kling-3-0, veo-3-1, seedance-2-5, studio-video-v1.")
    aspect_ratio: Optional[str] = Field(default=None, description="16:9, 9:16 or 1:1.")
    resolution: Optional[str] = Field(default=None, description="480p, 720p, 1080p or 4K where supported.")
    duration: Optional[int] = Field(default=None, description="Seconds, one of the model's lengths (often 5 or 10).")
    audio: Optional[bool] = Field(default=None, description="Generate native sound, on models that offer it.")
    image: Optional[str] = Field(default=None, description="Start frame (https URL or local path) for image to video.")
    end_image: Optional[str] = Field(default=None, description="End frame, on models that support it.")


class TanvoVideoGeneration(_TanvoTool):
    """Generate a video on Tanvo (text to video or image to video).

    By default the tool waits for the clip (usually 1–8 minutes); set `wait=False` to return the run id at once and
    check it with `TanvoGetGeneration`.
    """

    name: str = "tanvo_video_generation"
    description: str = (
        "Generate a short video from a prompt, or animate a still `image`, on Tanvo's video models (Kling, Veo, Seedance, "
        "Wan, MiniMax), optionally with native sound. Returns the video URL, or a run id to check later."
    )
    args_schema: type[BaseModel] = VideoInput
    default_model: str = "studio-video-v1"
    wait: bool = True
    timeout: float = 600

    def _run(self, prompt: str, model: Optional[str] = None, aspect_ratio: Optional[str] = None, resolution: Optional[str] = None, duration: Optional[int] = None, audio: Optional[bool] = None, image: Optional[str] = None, end_image: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            g = self._tanvo().generate_video(prompt, model=model or self.default_model, aspect=aspect_ratio, resolution=resolution, duration=duration, audio=audio, image=image, end_image=end_image, wait=self.wait, timeout=self.timeout)
        except tanvo.TanvoError as e:
            raise self._error(e, model or self.default_model) from e
        return self._result(g)


# ─── Songs ───────────────────────────────────────────────────────────────────


class MusicInput(BaseModel):
    prompt: str = Field(description="Describe mode: one sentence about the song. Lyrics mode: the lyrics with [Verse] / [Chorus] tags. Instrumental: the sound.")
    mode: Literal["Describe", "Lyrics", "Instrumental"] = Field(default="Describe")
    style: Optional[str] = Field(default=None, description="Genre, instruments, tempo, voice, e.g. 'acoustic pop, warm female vocal'.")
    title: Optional[str] = Field(default=None, description="Song title (Lyrics and Instrumental modes).")


class TanvoMusicGeneration(_TanvoTool):
    """Write a song on Tanvo (Suno V6): two takes, each an MP3 with cover art, a title and the lyrics. Needs an API key."""

    name: str = "tanvo_music_generation"
    description: str = "Write and sing a song (two takes, MP3 with cover art and lyrics) from a description, your own lyrics, or as an instrumental."
    args_schema: type[BaseModel] = MusicInput

    def _run(self, prompt: str, mode: str = "Describe", style: Optional[str] = None, title: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            g = self._tanvo().generate_music(prompt, mode=mode, style=style, title=title)
        except tanvo.TanvoError as e:
            raise self._error(e) from e
        return self._result(g)


# ─── Ready-made apps ─────────────────────────────────────────────────────────

_STOP = {"a", "an", "the", "my", "me", "of", "to", "into", "in", "on", "for", "and", "with", "make", "turn", "ai", "generator", "photo", "picture", "image", "this", "it", "like", "look", "looks"}
_ALIAS = {"dog": ["pet"], "cat": ["pet"], "puppy": ["pet"], "kitten": ["pet"], "song": ["music"], "restore": ["restoration"], "toy": ["figurine"], "figure": ["figurine"], "king": ["royal"], "queen": ["royal"]}


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1 and w not in _STOP]


class FindAppsInput(BaseModel):
    query: str = Field(description="What the user wants, e.g. 'royal pet portrait', 'action figure of me', 'restore an old photo', 'birthday song'.")
    kind: Optional[Literal["image", "video", "music"]] = Field(default=None, description="Only apps that make this kind of media.")


class TanvoFindApps(_TanvoTool):
    """Search Tanvo's 80 ready-made apps (pet portraits, figurines, photo restoration, trend videos, songs) by intent."""

    name: str = "tanvo_find_apps"
    description: str = (
        "Find a ready-made Tanvo app for what the user wants (pet portraits, figurines, old photo restoration, product shots, "
        "trend videos, birthday songs). Returns app slugs with their looks; run one with tanvo_run_app."
    )
    args_schema: type[BaseModel] = FindAppsInput
    limit: int = 6

    def _run(self, query: str, kind: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            apps = self._tanvo().apps(kind)
        except tanvo.TanvoError as e:
            raise self._error(e) from e
        terms = {t for w in _words(query) for t in [w, *_ALIAS.get(w, [])]}

        def score(a: tanvo.App) -> float:
            fields = [(a.name, 4), (a.slug, 3), (a.description, 2), (a.category or "", 2), (" ".join(l.name for l in a.looks), 2), (" ".join(l.prompt for l in a.looks), 0.5)]
            return sum(w for text, w in fields for t in terms if any(x == t or (len(t) > 3 and x.startswith(t[:-1])) for x in _words(text)))

        ranked = sorted((a for a in apps if score(a) > 0), key=score, reverse=True)[: self.limit]
        if not ranked:
            return f"No Tanvo app matched '{query}'. Use tanvo_image_generation or tanvo_video_generation with a written prompt.", {"apps": []}
        lines = [
            f"- {a.name} (app='{a.slug}', {a.kind}): {a.description} Photos needed: {sum(1 for i in a.inputs if i.min > 0)}. Looks: {', '.join(l.id for l in a.looks)}."
            for a in ranked
        ]
        return "Matching apps:\n" + "\n".join(lines), {"apps": [asdict(a) for a in ranked]}


class RunAppInput(BaseModel):
    app: str = Field(description="App slug from tanvo_find_apps.")
    look: Optional[str] = Field(default=None, description="Look id; defaults to the app's first look.")
    photos: list[str] = Field(default_factory=list, description="The user's photos, in the order the app needs them: https URLs or local paths.")
    details: Optional[str] = Field(default=None, description="A personal touch woven into the prompt, e.g. 'for my sister Ana'.")


class TanvoRunApp(_TanvoTool):
    """Run one of a Tanvo app's preset looks on the user's photos (uses the look's tuned prompt and model)."""

    name: str = "tanvo_run_app"
    description: str = "Run a Tanvo app's preset look on the user's photos, e.g. the 'royal' look of the pet portrait app. Returns the result URL."
    args_schema: type[BaseModel] = RunAppInput

    def _run(self, app: str, look: Optional[str] = None, photos: Optional[list[str]] = None, details: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            g = self._tanvo().generate_from_app(app, look=look, photos=photos or [], details=details)
        except (tanvo.TanvoError, KeyError, ValueError) as e:
            raise ToolException(str(e)) from e
        return self._result(g)


class GetGenerationInput(BaseModel):
    id: str = Field(description="Run id returned by another Tanvo tool.")


class TanvoGetGeneration(_TanvoTool):
    """Check a Tanvo run started earlier (videos take minutes)."""

    name: str = "tanvo_get_generation"
    description: str = "Check the status of a Tanvo run by id and get its file URLs once it has finished."
    args_schema: type[BaseModel] = GetGenerationInput

    def _run(self, id: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, dict[str, Any]]:
        try:
            g = self._tanvo().get(id)
        except tanvo.TanvoError as e:
            raise self._error(e) from e
        return self._result(g)


# ─── Toolkit ─────────────────────────────────────────────────────────────────


class TanvoToolkit(BaseToolkit):
    """All Tanvo tools sharing one client.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_tanvo import TanvoToolkit

        agent = create_agent("openai:gpt-5", tools=TanvoToolkit().get_tools())
        agent.invoke({"messages": [{"role": "user", "content": "Make my dog a royal portrait: https://…/rex.jpg"}]})
        ```
    """

    api_key: Optional[SecretStr] = Field(default_factory=secret_from_env("TANVO_API_KEY", default=None))
    base_url: Optional[str] = None
    video_wait: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_tools(self) -> list[BaseTool]:
        client = tanvo.Client(self.api_key.get_secret_value() if self.api_key else None, base_url=self.base_url)
        common = {"client": client}
        return [
            TanvoFindApps(**common),
            TanvoRunApp(**common),
            TanvoImageGeneration(**common),
            TanvoVideoGeneration(**common, wait=self.video_wait),
            TanvoMusicGeneration(**common),
            TanvoGetGeneration(**common),
        ]
