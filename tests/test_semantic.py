"""Tests for lightweight semantic retrieval."""
from __future__ import annotations

from weather_agents.core.semantic import SemanticScorer


class TestSemanticScorer:
    def setup_method(self):
        self.scorer = SemanticScorer()

    def test_identical_strings(self):
        score = self.scorer.similarity("hello world", "hello world")
        assert score == 1.0

    def test_completely_different(self):
        score = self.scorer.similarity("aaaaa", "bbbbb")
        assert score == 0.0

    def test_partial_overlap(self):
        score = self.scorer.similarity("deploy to server", "deployment script")
        assert 0.0 < score < 1.0

    def test_cjk_similarity(self):
        score = self.scorer.similarity("部署", "部署命令")
        assert score > 0.0

    def test_empty_strings(self):
        assert self.scorer.similarity("", "hello") == 0.0
        assert self.scorer.similarity("hello", "") == 0.0
        assert self.scorer.similarity("", "") == 0.0

    def test_case_insensitive(self):
        same = self.scorer.similarity("Hello World", "hello world")
        assert same == 1.0

    def test_mixed_language(self):
        score = self.scorer.similarity("pnpm install", "pnpm 安装")
        assert score > 0.0

    def test_code_identifiers(self):
        score = self.scorer.similarity("snake_case_var", "snakeCaseVar")
        assert score > 0.0

    def test_rank_returns_ordered(self):
        candidates = [
            {"key": "k1", "value": "deploy to production"},
            {"key": "k2", "value": "install dependencies"},
            {"key": "k3", "value": "rollback version"},
        ]
        ranked = self.scorer.rank("deploy", candidates, key_field="value", top_k=2)
        assert len(ranked) <= 2
        assert ranked[0][1]["key"] == "k1"  # most relevant first

    def test_rank_filters_below_min_score(self):
        candidates = [
            {"key": "k1", "value": "completely unrelated text here"},
        ]
        ranked = self.scorer.rank("zzzzz", candidates, key_field="value", min_score=0.5)
        assert len(ranked) == 0

    def test_rank_uses_key_field_boost(self):
        candidates = [
            {"key": "deploy_command", "value": "npm run build"},
        ]
        ranked = self.scorer.rank("deploy", candidates, key_field="value", top_k=1)
        assert len(ranked) == 1

    def test_get_scorer_singleton(self):
        from weather_agents.core.semantic import get_scorer

        s1 = get_scorer()
        s2 = get_scorer()
        assert s1 is s2


class TestFingerprintCaching:
    def test_cache_hits(self):
        scorer = SemanticScorer()
        fp1 = scorer._fingerprint("hello world" * 20)
        fp2 = scorer._fingerprint("hello world" * 20)
        assert fp1 is fp2  # same object from cache
