from __future__ import annotations

from pathlib import Path

import pytest

from init.io import resolve_scene_root


def test_scene_root_uses_override_before_config() -> None:
    assert resolve_scene_root(
        {"scene": {"root": "configured"}}, Path("overridden")
    ) == Path("overridden")


@pytest.mark.parametrize("config", [{}, {"scene": {}}, {"scene": {"root": ""}}])
def test_scene_root_is_required(config) -> None:
    with pytest.raises(ValueError, match=r"pass --scene-root or set scene\.root"):
        resolve_scene_root(config)
