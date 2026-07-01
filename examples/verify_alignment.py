"""Independently verify that the tool's alignment diagnostic is correct on real data.

It recomputes the reward-vs-human rank correlation FROM SCRATCH (a tie-averaged
Spearman, implemented here by hand) and checks it matches the number the tool's
own ``run_alignment`` reports. If they agree, the diagnostic is trustworthy on
this dataset — not just on the synthetic demo.

Run:  python3 examples/verify_alignment.py \
         --rubric examples/helpfulness_rubric.yaml \
         --responses examples/helpsteer2_responses.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rubric_reward_lens.diagnostics.alignment import run_alignment
from rubric_reward_lens.grader import FakeGrader
from rubric_reward_lens.models import Response, Rubric


def avg_rank(a: list[float]) -> np.ndarray:
    """Ranks with ties broken by AVERAGING (what a correct Spearman does).
    A naive argsort-rank gives the wrong answer when rewards are tied."""
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def tie_averaged_spearman(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(avg_rank(x), avg_rank(y))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", default="examples/helpfulness_rubric.yaml")
    ap.add_argument("--responses", default="examples/helpsteer2_responses.json")
    args = ap.parse_args()

    rubric = Rubric.from_yaml(args.rubric)
    data = json.load(open(args.responses, encoding="utf-8"))
    responses = [
        Response(id=r["id"], text=r["text"], prompt=r.get("prompt", ""), human_score=r["human_score"])
        for r in data
    ]

    grader = FakeGrader()

    # --- our independent recompute ---
    rewards = [grader.grade(rubric, r).reward for r in responses]
    hv = [r.human_score for r in responses]
    lo, hi = min(hv), max(hv)
    humans = [(v - lo) / (hi - lo) for v in hv]  # mirror the tool's [0,1] normalisation
    ours = tie_averaged_spearman(rewards, humans)

    # --- the tool's own number ---
    tool = run_alignment(rubric, grader, responses).correlation

    print(f"responses               : {len(responses)} real, human-labelled")
    print(f"independent Spearman     : {ours:.4f}")
    print(f"tool run_alignment()     : {tool:.4f}")
    match = abs(ours - tool) < 5e-3
    print("RESULT                   :", "PASS ✅ tool matches independent recompute" if match else "FAIL ❌ mismatch")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
