"""Quickstart: audit the bundled synthetic reward and print the verdict.

Run from the repo root with:  python3 examples/quickstart.py
(works from a clone without installing; after `pip install` the shim is a no-op)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rubric_reward_lens import FakeGrader, audit, load_demo  # noqa: E402

rubric, responses = load_demo()

# FakeGrader is a deterministic, keyword-presence grader — intentionally
# hackable, so the audit will flag it. Swap in OpenRouterGrader(model=...) to
# audit a real LLM grader.
card = audit(rubric, FakeGrader(), responses, human_labels=True)

print(card.to_markdown())
print("VERDICT:", card.verdict)
