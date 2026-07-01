import json

from rubric_reward_lens.diagnostics.alignment import AlignmentResult
from rubric_reward_lens.diagnostics.hacking import HackingResult
from rubric_reward_lens.diagnostics.monotonicity import MonotonicityResult
from rubric_reward_lens.diagnostics.order import OrderResult
from rubric_reward_lens.diagnostics.stability import StabilityResult
from rubric_reward_lens.diagnostics.structure import StructureResult
from rubric_reward_lens.report import ReportCard, _what_it_means


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


def test_what_it_means_hacking():
    not_hackable = HackingResult(0.0, (0.0, 0.0), {"keyword_stuff": (0.0, 0.0, 0.0)}, {"c1": 0.0}, False, 4)
    assert _what_it_means("hacking", not_hackable) == "not gameable"
    hackable = HackingResult(0.5, (0.3, 0.7), {"keyword_stuff": (0.5, 0.3, 0.7)}, {"leaky": 0.5, "safe": 0.0}, True, 4)
    msg = _what_it_means("hacking", hackable)
    assert "gameable via keyword_stuff" in msg and "leaky" in msg and "safe" not in msg


def test_what_it_means_monotonicity():
    clean = MonotonicityResult(-0.9, (-1.0, -0.8), 0, 0.6, True, 4)
    assert _what_it_means("monotonicity", clean) == "reward falls cleanly as quality drops"
    mostly = MonotonicityResult(-0.7, (-0.9, -0.5), 3, 0.3, True, 4)
    assert _what_it_means("monotonicity", mostly) == "mostly tracks quality (3 inversions)"
    bad = MonotonicityResult(0.1, (-0.2, 0.4), 5, 0.1, False, 4)
    assert _what_it_means("monotonicity", bad) == "does not reliably track quality (5 inversions)"


def test_what_it_means_stability():
    assert _what_it_means("stability", StabilityResult(0.0, 0.0, {}, True, 4)) == "identical on re-grade"
    assert _what_it_means("stability", StabilityResult(0.3, 0.1, {}, False, 4)) == "wobbles across re-grades"


def test_what_it_means_structure():
    clean = StructureResult([], {}, {"c1": 0.5}, [], 4)
    assert _what_it_means("structure", clean) == "all criteria informative"
    low = StructureResult([], {}, {"a": 1.0, "b": 0.0}, ["a", "b"], 4)
    assert _what_it_means("structure", low) == "2 low-signal: a, b"


def test_what_it_means_criterion_order():
    inv = OrderResult(0.0, 0.0, 0.0, True, 4)
    assert _what_it_means("criterion_order", inv) == "order-invariant"
    drift = OrderResult(0.3, 0.5, 0.5, False, 4)
    assert _what_it_means("criterion_order", drift) == "reward drifts with criterion order (mean Δ 0.30)"


def test_report_includes_criterion_order():
    order = OrderResult(mean_drift=0.3, max_drift=0.5, flip_rate=0.5, order_invariant=False, n=4)
    card = ReportCard("r", 4, order=order)
    md = card.to_markdown()
    assert "criterion_order" in md and "drifts with criterion order" in md
    assert abs(card.sub_scores()["criterion_order"] - 0.70) < 1e-9  # 1 - 0.30
    data = json.loads(card.to_json())
    assert data["diagnostics"]["criterion_order"]["mean_drift"] == 0.3


def test_what_it_means_alignment():
    assert _what_it_means("alignment", AlignmentResult(0.9, 0.8, 0.7, 0.1, (0.6, 1.0), 4)) == "matches human scores"
    assert _what_it_means("alignment", AlignmentResult(0.3, 0.4, 0.3, 0.2, (0.1, 0.6), 4)) == "weak agreement with humans"
    assert _what_it_means("alignment", AlignmentResult(0.0, 0.0, 0.0, 0.5, (-0.2, 0.2), 4)) == "does not match human scores"


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


def test_to_markdown_single_diagnostics_table():
    card = ReportCard("r", 4, hacking=_hackable(), monotonicity=_robust_mono())
    md = card.to_markdown()
    assert "# Reward Report Card" in md
    assert "## Diagnostics" in md and "What it means" in md
    assert "higher is better" in md
    # raw detail sections are gone
    assert "## Reward hacking" not in md
    assert "## Discrimination" not in md
    # points to JSON for raw metrics
    assert "report.json" in md


def test_to_markdown_names_leak_in_plain_english_not_raw_numbers():
    # _hackable() has per_criterion_gameability={"c1": 0.5}, overall 0.4
    md = ReportCard("r", 4, hacking=_hackable()).to_markdown()
    assert "gameable" in md and "c1" in md          # leak named in words
    assert "0.400" not in md and "1.000" not in md  # no raw gameability/gain numbers
    # JSON still carries the raw dict
    data = json.loads(ReportCard("r", 4, hacking=_hackable()).to_json())
    assert "per_criterion_gameability" in data["diagnostics"]["hacking"]


def test_minor_gaming_with_high_trust_says_caution_not_fix():
    # small but significant gain + otherwise-healthy diagnostics -> the headline
    # must NOT scream "fix before training"; it should be a softer caution.
    minor = HackingResult(
        overall_hack_gain=0.07,
        ci=(0.02, 0.12),
        per_probe={"keyword_stuff": (0.07, 0.02, 0.12)},
        per_criterion_gameability={"c1": 0.07},
        hackable=True,
        n=4,
    )
    card = ReportCard("r", 4, hacking=minor, monotonicity=_robust_mono())
    assert card.trust_score >= 0.6
    assert "Fix the rubric before training" not in card.verdict
    assert "Caution" in card.verdict


def test_to_html_is_html(tmp_path):
    card = ReportCard("r", 4, hacking=_hackable())
    out = tmp_path / "card.html"
    doc = card.to_html(str(out))
    assert doc.startswith("<!doctype html>") and "Hackable" in doc
    assert out.read_text().startswith("<!doctype html>")
