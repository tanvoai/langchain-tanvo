"""Live test against tanvo.ai (one free image, about $0.005 upstream). Run with: TANVO_LIVE=1 pytest tests/integration_tests"""

import os

import pytest

from langchain_tanvo import TanvoFindApps, TanvoImageGeneration

pytestmark = pytest.mark.skipif(os.environ.get("TANVO_LIVE") != "1", reason="set TANVO_LIVE=1 to call tanvo.ai")


def test_find_apps_live() -> None:
    msg = TanvoFindApps().invoke({"type": "tool_call", "name": "tanvo_find_apps", "id": "1", "args": {"query": "royal pet portrait"}})
    assert "ai-pet-portrait-generator" in msg.content


def test_free_image_live() -> None:
    msg = TanvoImageGeneration().invoke({"type": "tool_call", "name": "tanvo_image_generation", "id": "1", "args": {"prompt": "a paper boat on a puddle", "aspect_ratio": "4:3"}})
    assert msg.artifact["status"] == "SUCCEEDED" and msg.artifact["outputs"][0]["url"].startswith("https://")
