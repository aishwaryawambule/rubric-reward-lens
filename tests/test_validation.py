"""The tool's own falsification test.

Mirrors the qualitative finding of "Reproducing, Analyzing, and Detecting
Reward Hacking in Rubric-Based RL" (arXiv:2606.04923): a presence-style rubric
graded by a keyword-presence grader is gameable, while a grader that requires
real prose is not. If the tool cannot separate the two, it is not doing its job.

``rubric``, ``hackable_responses`` and ``robust_grader`` come from conftest.py.
"""

from __future__ import annotations

from rubric_reward_lens.audit import audit
from rubric_reward_lens.grader import FakeGrader


def test_tool_flags_hackable_reward(rubric, hackable_responses):
    bad = audit(rubric, FakeGrader(), hackable_responses)  # keyword-presence: stuffable
    assert bad.hacking.hackable is True
    assert "Hackable" in bad.verdict


def test_tool_clears_robust_reward(rubric, hackable_responses, robust_grader):
    good = audit(rubric, robust_grader, hackable_responses)
    assert good.hacking.hackable is False


def test_robust_reward_scores_higher_trust(rubric, hackable_responses, robust_grader):
    bad = audit(rubric, FakeGrader(), hackable_responses)
    good = audit(rubric, robust_grader, hackable_responses)
    assert good.trust_score > bad.trust_score
