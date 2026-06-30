import json

from rubric_reward_lens.diagnostics.hacking import HackingResult
from rubric_reward_lens.diagnostics.monotonicity import MonotonicityResult
from rubric_reward_lens.report import ReportCard


def _hackable():
    return HackingResult(
        overall_hack_gain=0.4,
        ci=(0.2, 0.6),
        per_probe={"keyword_stuff": (0.5, 0.3, 0.7)},
        per_criterion_gameability={"c1": 0.5},
        hackable=True,
        n=4,
    )


def _robust_mono():
    return MonotonicityResult(
        spearman=-0.95, ci=(-1.0, -0.8), inversions=0, separation=0.6, monotonic=True, n=4
    )


def test_hackable_card_low_trust_and_verdict():
    card = ReportCard("r", 4, hacking=_hackable())
    assert card.trust_score < 0.7
    assert "Hackable" in card.verdict


def test_robust_card_high_trust():
    card = ReportCard("r", 4, monotonicity=_robust_mono())
    assert card.trust_score > 0.7
    assert "Robust" in card.verdict


def test_to_json_roundtrips():
    card = ReportCard("r", 4, hacking=_hackable(), monotonicity=_robust_mono())
    data = json.loads(card.to_json())
    assert data["rubric_name"] == "r"
    assert "trust_score" in data and "diagnostics" in data


def test_to_markdown_has_sections():
    card = ReportCard("r", 4, hacking=_hackable())
    md = card.to_markdown()
    assert "# Reward Report Card" in md and "Reward hacking" in md


def test_to_html_is_html(tmp_path):
    card = ReportCard("r", 4, hacking=_hackable())
    out = tmp_path / "card.html"
    doc = card.to_html(str(out))
    assert doc.startswith("<!doctype html>") and "Hackable" in doc
    assert out.read_text().startswith("<!doctype html>")
