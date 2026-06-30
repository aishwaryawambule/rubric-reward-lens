# rubric-reward-lens

**A pre-flight auditor for rubric-based LLM rewards.** Point it at your rubric, your
grader, and a few example answers — it tells you, *before you spend GPU on RL*, whether
the reward signal actually measures quality or can be gamed.

> ⚠️ Status: v0.1, early. The API may change. Issues and PRs welcome.

## The problem

To train an LLM on a "soft" task with no single right answer (long-form writing, medical
Q&A, support replies), you need a **score** for every attempt. The popular recipe —
[**Rubrics as Rewards** (Gunjal et al., 2025, arXiv:2507.17746)](https://arxiv.org/abs/2507.17746)
— is to write a checklist (a *rubric*), have an LLM grade each answer against it, and use
that grade as the reward.

That reward is fragile and **fails silently**. Recent work shows it gets *hacked*:

- [Reward Hacking in Rubric-Based RL (arXiv:2605.12474)](https://arxiv.org/abs/2605.12474) —
  gains concentrate in presence/completeness criteria ("did it mention X") while factual
  correctness, conciseness, and relevance *decline*; stronger graders don't fix an
  under-specified rubric.
- [Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based RL (arXiv:2606.04923)](https://arxiv.org/abs/2606.04923) —
  characterizes how to *detect* it.

You usually discover the reward was gameable **only after** an expensive RL run produces a
worse model that scores higher. `rubric-reward-lens` turns that detection into an
off-the-shelf, `pip install`-able check you run first.

It audits a **different object** from the reward-model benchmarks
([RewardBench 2, arXiv:2506.01937](https://arxiv.org/abs/2506.01937);
[RM-Bench, arXiv:2410.16184](https://arxiv.org/abs/2410.16184)): those score *learned
reward models* on preference accuracy. This scores a **rubric + LLM-grader reward signal
for gameability**.

## Install

```bash
pip install rubric-reward-lens   # (or: pip install -e . from a clone)
```

Inference-only: needs `numpy`, `scipy`, `pyyaml`, `httpx`. No GPU, no `torch`.

## Quickstart

```python
from rubric_reward_lens import Rubric, OpenRouterGrader, audit, load_demo

# Bring your own, or try the bundled synthetic demo:
rubric, responses = load_demo()

grader = OpenRouterGrader(model="anthropic/claude-haiku-4.5")  # needs OPENROUTER_API_KEY
card = audit(rubric, grader, responses, human_labels=True)

print(card.verdict)            # e.g. "⚠️ Hackable — reward can be gamed for +0.34 ..."
card.to_html("report.html")    # full report card
```

No API key? The CLI runs the whole pipeline offline with a deterministic grader:

```bash
rrl demo --out report.html
rrl audit --rubric examples/rubric.yaml --grader examples/grader.fake.yaml \
          --responses responses.json --out report.html
```

## What it checks (the diagnostics)

All diagnostics are **label-free** except the last; bootstrap confidence intervals are
reported throughout (following
[Adding Error Bars to Evals, arXiv:2411.00640](https://arxiv.org/abs/2411.00640)).

| Diagnostic | Question it answers |
| --- | --- |
| **Reward hacking** | Do cheap "wins" — keyword-stuffing, verbosity padding, confident-wrong claims, format mimicry — earn reward they shouldn't? |
| **Discrimination / monotonicity** | When an answer is progressively degraded, does the reward actually fall? |
| **Grader stability** | Re-grading the same answer, how much does the reward wobble? |
| **Criterion structure** | Are criteria redundant, low-signal, or does the reward over-depend on one? |
| **Human alignment** *(optional)* | When you have human scores, how well does the reward agree (QWK / κ / calibration)? |

The report card combines these into a composite **trust score** and a one-line verdict —
without hiding the per-diagnostic breakdown.

## Using an LLM-as-a-judge for anything?

You don't need to be doing RL. If you grade *anything* with an LLM against a rubric
(evals, autograding, quality gates), the same machinery tells you whether that judge can
be gamed and how stable it is. Use `audit(...)` with your own `Grader`.

## How it's validated

The tool has its own falsification test (`tests/test_validation.py`): it must flag a
presence-only rubric graded by a keyword-matching grader as **hackable**, and clear a
grader that resists the same probes as **robust** — reproducing the qualitative pattern of
arXiv:2606.04923. If it can't separate the two, it doesn't ship.

## Note on the demo data

The bundled `healthbench_demo` dataset is **synthetic and hand-authored** — inspired by the
*shape* of [HealthBench (OpenAI, arXiv:2505.08775)](https://arxiv.org/abs/2505.08775)
(physician-style rubric + graded responses + human scores), but **not** real HealthBench
data, to avoid any licensing question. Bring your own data to audit a real reward.

## License

MIT
