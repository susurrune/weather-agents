"""Tests for agent icon system."""

from __future__ import annotations


class TestAgentColorMap:
    def test_all_agents_have_colors(self):
        from weather_agents.core.icons import AGENT_COLOR_MAP

        for name in ("fog", "rain", "frost", "snow", "dew", "fair"):
            assert name in AGENT_COLOR_MAP


class TestSvgPath:
    def test_returns_string(self):
        from weather_agents.core.icons import svg_path

        path = svg_path("fog")
        assert isinstance(path, str)
        assert path.endswith("fog.svg")

    def test_returns_absolute_path(self):
        from weather_agents.core.icons import svg_path

        path = svg_path("rain")
        assert "\\" in path or "/" in path
        assert "icons" in path


class TestIconText:
    def test_returns_correct_symbol(self):
        from weather_agents.core.icons import icon_text

        assert icon_text("fog") == "≋"
        assert icon_text("rain") == "⸽"
        assert icon_text("frost") == "✱"
        assert icon_text("snow") == "❉"
        assert icon_text("dew") == "∘"
        assert icon_text("fair") == "☼"

    def test_returns_name_for_unknown(self):
        from weather_agents.core.icons import icon_text

        assert icon_text("unknown") == "unknown"
