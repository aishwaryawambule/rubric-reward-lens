# rubric-reward-lens — Design Spec

> A pre-flight auditor for rubric-based LLM rewards. Point it at your rubric + grader +
> a few example answers, and it tells you — **before you spend GPU on RL** — whether the
> reward signal actually measures quality or can be gamed.

*Status: draft for review. Date: 2026-06-29.*

---

## 1. Problem & motivation

Reinforcement learning on "soft" tasks (long-form writing, medical Q&A, support replies)
has no verifiable ground truth, so a reward must be manufactured. The dominant recipe —
**Rubrics as Rewards (RaR)** ([2507.17746](https://arxiv.org/abs/2507.17746)) — is: write a
rubric (a weighted checklist of criteria), have an LLM grade each model response against it,
and use the aggregated grade as the scalar reward the policy maximizes.

This reward is **fragile, and fails silently**. Two recent papers characterize the failure:

- [**Reward Hacking in Rubric-Based RL (2605.12474)**](https://arxiv.org/abs/2605.12474) —
  gains concentrate in *presence/completeness* criteria ("did it mention X") while
  *factual correctness, conciseness, relevance* decline. Stronger graders do **not** prevent
  this when the rubric under-specifies failure modes.
- [**Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based RL (2606.04923)**](https://arxiv.org/abs/2606.04923)
  — June 2026 — is about **detecting** the hack.

The cost is paid late: you discover the reward was gameable only after an expensive RL run
produced a worse model that scores higher.

**Prior art audits a different object.** [RewardBench 2 (2506.01937)](https://arxiv.org/abs/2506.01937)
and [RM-Bench (2410.16184)](https://arxiv.org/abs/2410.16184) grade *learned reward models* on
*preference-pair accuracy*. They do not take a rubric + LLM-grader reward and ask "can this be
gamed?"

**The gap:** the research says this failure happens and sketches how to detect it, but there is
no installable tool that runs the detection on your own setup. `rubric-reward-lens` is that tool.

One-line framing: **it turns the "detecting reward hacking" finding of
[2606.04923](https://arxiv.org/abs/2606.04923) into an off-the-shelf, `pip install`-able
pre-flight check.**

---

## 2. Goals & non-goals

**Goals (v1)**
- A Python **library + CLI**, **inference-only** (no GPU, no training, no `torch`).
- **Bring-your-own** rubric + grader + responses; ship a **bundled demo** on a HealthBench subset.
- Produce a **Reward Report Card**: a headline verdict ("safe to optimize?" / "hackable")
  plus per-criterion breakdowns, in HTML + markdown + JSON.
- **Label-free diagnostics are the headline** (work with zero human labels); a human-anchored
  mode is optional and runs when labels exist.
- **Statistical rigor throughout** — bootstrap confidence intervals and significance on every
  headline number (per [2411.00640](https://arxiv.org/abs/2411.00640)).
- **Provider-agnostic** grader via OpenRouter / any OpenAI-compatible endpoint.

**Non-goals (v1)**
- No RL training loop (we audit the reward; we don't optimize against it).
- No learned reward-model evaluation — that is RewardBench/RM-Bench's job.
- No pairwise / position-bias diagnostics — RaR rewards are pointwise scalars; deferred to v2.
- No multimodal (text only).
- Not a new human-labeled benchmark; this is a **harness** you run on your data.

---

## 3. Core concepts & data model

- **`Criterion`** — `{ id, text, weight, polarity }` where `polarity ∈ {include, avoid}`
  ("response should include X" vs. "should avoid X").
- **`Rubric`** — an ordered list of `Criterion`, loadable from YAML/JSON.
- **`Response`** — `{ id, prompt_id, prompt, text, human_score? }` (human label optional).
- **`Grader`** — a callable `(rubric, response) → GraderResult`. Backed by an LLM (config:
  model, prompt template, temperature). The grading→scalar **aggregation** is pluggable
  (default: weighted sum of per-criterion scores, normalized to `[0,1]`).
- **`GraderResult`** — `{ per_criterion: [{criterion_id, score, justification}], reward }`.
- **`Probe`** — a transformation `Response → Response'` with a **known expected reward
  direction** (e.g. "this variant is strictly worse, reward should drop").
- **`ReportCard`** — the structured output: headline verdict, sub-scores, per-criterion tables,
  and the raw evidence behind each.

**Testability note:** `Grader` is an interface. A deterministic `FakeGrader` (rule-based, no
API) backs the unit tests so the suite runs offline and fast.

---

## 4. The diagnostics (the heart of the tool)

### Group A — label-free (the headline value)

**A1. Reward-hacking probes.** For each response, manufacture "cheap-win" variants that should
*not* earn much reward:
- **keyword-stuffing** — inject the rubric's expected phrases/facts with no real substance;
- **verbosity padding** — add fluff length without information;
- **confident-but-wrong** — assert plausible-sounding but incorrect claims on the criterion's topic;
- **format-mimicry** — adopt the shape of a good answer (headings, lists) without the content.

Metric: **hack gain** = `reward(variant) − reward(original)`, with per-criterion attribution
(which criteria the hack exploited). A positive hack gain on a quality-degraded variant means the
reward is hackable. Output a **gameability score** per criterion and overall, with bootstrap CIs.

**A2. Discrimination / monotonicity.** Build a **degradation ladder** from a strong answer
(progressively drop facts, introduce errors, truncate). The reward should be monotonically
non-increasing down the ladder. Metrics: Spearman correlation between corruption level and reward;
count of **inversions**; and a **separation** metric (reward gap between best and worst rungs).
A reward that can't rank an obviously-worse answer lower is unusable.

**A3. Grader stability.** Re-grade across seeds/temperature and with **paraphrased rubric
wording**. Metrics: per-criterion **flip rate**, reward **variance**, CI width. An unstable reward
is a noisy training signal (motivated by the prompt-sensitivity literature,
[2509.01790](https://arxiv.org/abs/2509.01790)).

### Group B — criterion-level structure (label-free, statistical)

**B1. Redundancy** — correlation matrix of per-criterion pass patterns across responses; flag
criteria that always co-fire (they silently over-weight one dimension).
**B2. Weight sensitivity** — how much the reward *ranking* changes if a criterion is
dropped/reweighted; flag criteria the reward over-depends on.
**B3. Coverage / discrimination** — per-criterion pass-rate distribution; criteria that always
pass or always fail carry no signal.

### Group C — human alignment (optional; runs only when labels exist)

**C1. Grader-vs-human agreement** — QWK / Cohen's κ / correlation between reward (overall and
per-criterion) and human scores. On HealthBench, report against the **doctor-doctor ceiling**.
**C2. Calibration** — does reward magnitude track human quality (reliability diagram + calibration
error).

**Statistical layer (cross-cutting):** bootstrap CIs (seeded) on every headline metric; paired
significance tests for comparisons (e.g. "is hack gain significantly > 0?").

---

## 5. Output — the Reward Report Card

- **Headline verdict**, e.g.
  `⚠️ Hackable — keyword-stuffing gains +0.34 reward (95% CI 0.21–0.47) on criteria C2, C5`
  or `✅ Robust — no probe earns significant reward; ladder monotonic (ρ = −0.97)`.
- **Sections**: gameability (per criterion), discrimination, stability, criterion structure,
  and — if labels were supplied — human alignment.
- **A single composite "reward trust score"** with documented, configurable sub-score weights.
- **Formats**: HTML (Jinja2) + markdown + machine-readable JSON.

---

## 6. API surface

**Library**
```python
from rubric_reward_lens import Rubric, Grader, load_responses, audit

rubric    = Rubric.from_yaml("rubric.yaml")
grader    = Grader.openrouter(model="anthropic/claude-haiku-4.5", prompt="grader.txt")
responses = load_responses("responses.jsonl")

report = audit(
    rubric, grader, responses,
    diagnostics=["hacking", "monotonicity", "stability", "structure"],
    human_labels=None,          # supply to enable Group C
)
report.to_html("card.html")
report.to_json("card.json")
print(report.verdict)
```

**CLI**
```
rrl audit --rubric rubric.yaml --grader grader.yaml --responses responses.jsonl --out card.html
rrl demo            # runs the full audit on the bundled HealthBench subset
```

---

## 7. How we prove the tool works (validation)

The tool gets its own **falsification test** (mirroring the evidence-first, match-beats-mismatch
discipline):

1. Construct, for one task, a **known-bad rubric** (under-specified, presence-only) and a
   **known-good rubric** (covers correctness + conciseness, penalizes the hacks).
2. Show the tool rates the bad rubric **hackable** (high, significant hack gain) and the good one
   **robust** — reproducing the qualitative pattern of [2606.04923](https://arxiv.org/abs/2606.04923).
3. Show monotonicity flags a grader that cannot rank a degradation ladder.

If the tool can't separate a known-bad from a known-good rubric, it's not shipping.

---

## 8. Tech stack

- Python 3.11+, `pyproject.toml`, `pytest`, `ruff`. **No GPU / no torch.**
- LLM access: OpenRouter (OpenAI-compatible SDK); API key from env, never committed.
- Stats: `numpy` / `scipy` (bootstrap, Spearman, κ).
- Report: `Jinja2` for HTML; stdlib for markdown/JSON.
- **TDD**: every diagnostic has unit tests backed by a deterministic `FakeGrader`, so the suite
  runs offline with no API calls; a thin integration test exercises one real grader behind a flag.
- Caching of grader calls (keyed by model+rubric+text) to control cost across re-grades.

---

## 9. Milestones

- **M0 — Scaffold + data spine.** Package, config, OpenRouter grader client, data models,
  bundled HealthBench-subset loader, `Grader` interface + `FakeGrader`.
- **M1 — Reward harness + report-card skeleton.** Run grader over a set, persist JSON, render an
  HTML/markdown card with a placeholder headline.
- **M2 — Label-free core (A1 hacking + A2 monotonicity).** The headline value.
- **M3 — Criterion structure (B) + stability (A3) + the statistical layer.**
- **M4 — Human-alignment mode (C).** Runs out-of-the-box on the HealthBench labels.
- **M5 — Proof + polish.** The §7 falsification test, CLI, README, a showcase report card, docs.

M0–M2 is the usable core; M3–M5 makes it credible and adoptable.

---

## 10. Risks, caveats & open questions

- **Probe quality.** Cheap-win variants must be convincingly "hacky but low quality." Risk: weak
  probes under-detect. Mitigation: combine templated probes with LLM-generated ones, and surface
  human-inspectable examples in the report so probe quality is auditable.
- **Grader cost.** Diagnostics re-grade many variants. Mitigation: caching, small default samples,
  async batching, and a `--budget` cap.
- **Monotonicity assumption.** The degradation ladder assumes corruptions truly lower quality;
  keep corruptions unambiguous and documented.
- **Composite trust-score weighting** is a judgment call — keep sub-scores first-class and the
  composite configurable; never hide the breakdown.
- **HealthBench bundling / licensing.** Verify the license before shipping any data subset; if
  unclear, ship a **loader + fetch script** rather than the data itself.

**Open questions for you**
1. **Name** — `rubric-reward-lens` / `rewardlens` / `rrl` / something else?
2. **Framing breadth** — keep it tightly aimed at *rubric-RL reward auditing*, or also present it
   to the broader *LLM-as-a-judge* crowd (same machinery, bigger audience)?
3. **Composite score** — ship a single headline trust score, or only the sub-scores in v1?
