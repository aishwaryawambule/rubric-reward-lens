# rubric-reward-lens

**A pre-flight auditor for rubric-based LLM rewards.** Point it at your rubric, your
grader, and a few example answers — it tells you, *before you spend GPU on RL*, whether
the reward signal actually measures quality or can be gamed.

> ⚠️ Status: v0.1, early. The API may change.

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

Not on PyPI yet — install from a clone:

```bash
git clone https://github.com/rubric-reward-lens/rubric-reward-lens
cd rubric-reward-lens
pip install .          # add ".[dev]" to also get pytest
```

Use a regular `pip install .`, **not** the editable `pip install -e .` — on recent
Python (3.14) the editable install can silently produce a non-importable package.
Once published, `pip install rubric-reward-lens` will work directly.

Requires Python ≥ 3.11. Inference-only: needs `numpy`, `pyyaml`, `httpx`. No GPU, no `torch`.

## Quickstart

Two ways to use it — as a **Python library** (embed it in an eval pipeline / CI gate / RL
setup) or from the **CLI** (quick one-off audits). Both are shown below; see
[**Usage**](docs/usage.md) for the full guide, custom graders, and integration patterns.

```python
from rubric_reward_lens import Rubric, OpenRouterGrader, audit, load_demo

# Bring your own, or try the bundled synthetic demo:
rubric, responses = load_demo()

grader = OpenRouterGrader(model="anthropic/claude-haiku-4.5")  # needs OPENROUTER_API_KEY
card = audit(rubric, grader, responses, human_labels=True)

print(card.verdict)            # e.g. "⚠️ Hackable — reward can be gamed for +0.34 ..."
card.to_html("report.html")    # full report card
```

Prefer a **local model** (no API key, no cost, no data leaves your machine)? Use a model
served by [Ollama](https://ollama.com):

```python
from rubric_reward_lens import OllamaGrader, audit, load_demo

rubric, responses = load_demo()
grader = OllamaGrader(model="qwen2.5:14b")   # needs `ollama serve` + the model pulled
card = audit(rubric, grader, responses, human_labels=True)
```

No grader at all? The CLI runs the whole pipeline offline with a deterministic grader:

```bash
rrl demo --out report.html
rrl audit --rubric examples/rubric.yaml --grader examples/grader.fake.yaml \
          --responses responses.json --out report.html
```

Grader configs for the CLI: `{type: fake}`, `{type: ollama, model: ...}`, or
`{type: openrouter, model: ...}` — see [examples/](examples/).

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
| **Criterion-order invariance** | Does the reward change when the rubric's criteria are listed in a different order? (judge position bias, [arXiv:2602.02219](https://arxiv.org/abs/2602.02219)) |
| **Human alignment** *(optional)* | When you have human scores, how well does the reward agree (QWK / κ / calibration)? |

The report card combines these into a composite **trust score** and a one-line verdict,
over a per-diagnostic table where every score reads 0–1 (1 = good) with a plain-English
"what it means" — raw metrics are kept in the JSON output:

```
⚠️ Caution — trust score 0.63; review the diagnostics before training.
- Composite trust score: 0.63  (0–1, higher is better)

## Diagnostics
| Diagnostic   | Score | What it means                                |
| hacking      | 0.97  | not gameable                                 |
| monotonicity | 0.69  | mostly tracks quality (7 inversions)         |
| stability    | 1.00  | identical on re-grade                        |
| structure    | 0.50  | 2 low-signal criteria: accurate, not_evasive |
| alignment    | 0.00  | does not match human scores                  |
```

See a full generated report (the output of `rrl demo`):
[**`sample_report.md`**](examples/sample_report.md) (renders right here on GitHub) or
[`sample_report.html`](examples/sample_report.html) (styled — open in a browser).

Two short docs explain the output: [**Interpreting the report card**](docs/interpreting-the-report.md)
(how to read it and decide if a reward is safe to train on) and
[**Metrics & scores reference**](docs/metrics.md) (the precise definition of every metric and score).

## Using an LLM-as-a-judge for anything?

You don't need to be doing RL. If you grade *anything* with an LLM against a rubric
(evals, autograding, quality gates), the same machinery tells you whether that judge can
be gamed and how stable it is. Use `audit(...)` with your own `Grader`.

## How it's validated

The tool has its own falsification test (`tests/test_validation.py`): it must flag a
presence-only rubric graded by a keyword-matching grader as **hackable**, and clear a
grader that resists the same probes as **robust** — reproducing the qualitative pattern of
arXiv:2606.04923. If it can't separate the two, it doesn't ship.

## Limitations

v0.1 is deliberately a small, honest core. Know these before you rely on it:

- **The four hack probes are not exhaustive.** They model the *documented* failure modes
  (presence, verbosity, confidence, format). Domain-specific gaming — sycophancy,
  prompt-injection, self-preference, fabricated citations — is not covered. Add your own
  probe (a function `(response, rubric) -> response`) for those.
- **Custom probes and diagnostics work — in code, not from the CLI.**
  `audit(..., probes=[...], extra_diagnostics=[...])` accepts your own hack probes (each a
  `Probe` wrapping a `(response, rubric) -> response` transform) and your own diagnostics (a
  callable `(rubric, grader, responses) -> DiagnosticResult`). The **CLI** (`rrl audit`)
  still uses the built-ins only — a YAML config can't hold a Python function. If you drive
  the tool from a shell and need a custom probe/diagnostic, write a short script that calls
  `audit(...)`. (Named/registered probes for the CLI are on the roadmap.)
- **Diagnostics need a real sample.** With very few responses, hacking / monotonicity /
  structure / alignment are statistical artifacts (any 2 points force a correlation of ±1).
  Aim for **~15–20+** responses; grader **stability** and **criterion-order** are the
  exceptions — they measure the *judge*, so they're meaningful at any n.
- **Auditing an LLM judge is not free.** Each response costs **~20 grade calls** (probes +
  degradation ladder + stability re-grades + criterion-order permutations), so cost ≈
  `20 × responses × grader-speed`. On a large local model this is minutes-to-hours; use a
  fast/cheap grader or a smaller sample.
- **The trust score is a rough heuristic, not a calibrated probability.** It's a plain
  average of the sub-scores — `0.75` doesn't mean "75% safe" — and because hacking is only
  *one term* in that average, a **gameable reward can still show a middling trust score**.
  So don't rely on the composite alone. The **`Hackable` verdict is the single most
  important thing to check — always heed it**: it disqualifies a reward on its own, no
  matter how high the trust score. Read the per-diagnostic table for the rest.
- **Early and evolving (v0.1).** Not yet stable software — the API, report format, and
  verdict thresholds may change between versions, so pin the version if you depend on it.

See [ROADMAP.md](ROADMAP.md) for what's planned next.

## Note on the demo data

The bundled `healthbench_demo` dataset is **synthetic and hand-authored** — inspired by the
*shape* of [HealthBench (OpenAI, arXiv:2505.08775)](https://arxiv.org/abs/2505.08775)
(physician-style rubric + graded responses + human scores), but **not** real HealthBench
data, to avoid any licensing question. Bring your own data to audit a real reward.

## License

MIT
