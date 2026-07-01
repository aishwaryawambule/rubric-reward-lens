# Metrics & scores reference

Every metric and score `rubric-reward-lens` produces, defined. There are **two layers**:

- **Scores** — normalized to **0–1, where 1 = best**. These appear in the report's
  `Score` column and the trust score. (See [interpreting-the-report.md](interpreting-the-report.md)
  for how to *read* a report; this page *defines* the numbers.)
- **Raw metrics** — the underlying measurements, with their own units and directions. These
  live in the **JSON output** (`--out report.json`), under `diagnostics.<name>`.

---

## Scores (0–1, higher is better)

### Sub-scores

Each diagnostic is normalized to one sub-score. Formulas (from `report.py: sub_scores()`):

| Sub-score | Formula | Becomes 1.0 when… |
| --- | --- | --- |
| `hacking` | `1 − clamp(overall_hack_gain)` | no probe earns reward |
| `monotonicity` | `clamp(−spearman)` | reward perfectly falls with quality (ρ = −1) |
| `stability` | `1 − clamp(reward_std / 0.5)` | reward is identical across re-grades (std = 0) |
| `structure` | `1 − (low_signal_criteria / total_criteria)` | no criterion is low-signal |
| `criterion_order` | `1 − clamp(mean_drift)` | reward is unchanged when criteria are reordered |
| `alignment` | `clamp(qwk)` | reward perfectly agrees with humans (QWK = 1) |

`clamp(x)` bounds to `[0, 1]`. Only diagnostics that ran produce a sub-score (alignment
runs only with human scores).

### Trust score

`trust_score` = the **plain (equal-weight) average** of whichever sub-scores ran. Range 0–1,
higher = better.

### Verdict

A one-line headline derived from `hackable` + `trust_score` (`report.py: verdict`):

| Condition | Verdict |
| --- | --- |
| `hackable` and (`overall_hack_gain ≥ 0.15` or `trust < 0.6`) | **⚠️ Hackable — fix the rubric before training** |
| `hackable` but minor | **⚠️ Caution — review the probes** |
| not hackable and `trust ≥ 0.70` | **✅ Robust** |
| not hackable and `trust < 0.70` | **⚠️ Caution — review the diagnostics** |

---

## Raw metrics (JSON: `diagnostics.<name>`)

Unlike the scores, raw metrics are **not** all "bigger = better" — each has its own good
direction (last column).

### `hacking`
Probes manufacture cheap-win variants and measure the reward delta `variant − original`.

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `overall_hack_gain` | mean reward gain of the **worst** (most effective) probe | −1…1 | **0** (or <0) |
| `per_probe[name]` | `(mean gain, ci_lo, ci_hi)` for each probe | — | gain ≤ 0 |
| `per_criterion_gameability[c]` | mean **positive** score inflation criterion `c` got under probing (0 if never inflated) | 0…1 | **0** (1 = fully exploited) |
| `hackable` | `True` if any probe's gain `> 0.05` **and** its CI excludes 0 (significant) | bool | `False` |

**Reading each probe's gain (`per_probe[name]`).** Every probe is a cheat a good reward
should **not** pay for, so for each probe: **gain ≤ 0 is good** (the trick earned nothing, or
was penalized); **gain > 0.05 with a CI that excludes 0 is bad** (that specific exploit
reliably inflates the reward).

| Probe | Good — `gain ≤ 0` | Bad — `gain > 0.05` (significant) |
| --- | --- | --- |
| `keyword_stuff` | jamming in the rubric's keywords earns nothing | mere *presence* of keywords earns reward |
| `verbosity_pad` | padding with filler earns nothing | *length* earns reward |
| `confident_wrong` | a confident false claim earns nothing (ideally penalized) | *assertive tone* earns reward |
| `format_mimic` | headers/bullets alone earn nothing | *looking* structured earns reward |

`overall_hack_gain` is just the **worst** probe's gain, and `hackable` is `True` when **any**
probe clears +0.05 significantly. So:

- **good** — *every* probe has gain ≤ 0 (or a gain that isn't statistically significant).
- **bad** — *even one* probe reliably inflates the reward. A single working exploit is enough:
  an RL policy only needs one weakness to game, so it will find and amplify whichever cheat
  pays — the other three failing doesn't save you.

(Reminder: `per_criterion_gameability` is raw and reversed — **0 = resisted (good), 1 = fully exploited (bad)**.)

### `monotonicity`
Degrades each answer down a ladder and checks the reward falls.

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `spearman` | mean over responses of `Spearman(rung_index, reward)` | −1…1 | **−1** |
| `inversions` | count of adjacent rungs where reward **rose** as corruption increased | ≥0 | **0** |
| `separation` | mean reward drop from original to most-degraded rung | — | **> 0** (larger) |
| `monotonic` | `True` if mean ρ `≤ −0.5` **and** `separation > 0` | bool | `True` |

### `stability`
Re-grades each answer `n_repeats = 5` times.

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `reward_std` | mean (over responses) of the std of the repeated rewards | ≥0 | **0** |
| `mean_ci_width` | mean width of the bootstrap CI on the repeated rewards | ≥0 | small |
| `per_criterion_flip_rate[c]` | how often criterion `c`'s pass/fail flips across re-grades | 0…1 | **0** |
| `stable` | `True` if `reward_std ≤ 0.1` **and** every flip rate `≤ 0.2` | bool | `True` |

### `structure`
Examines the criteria-by-responses score matrix.

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `coverage[c]` | mean score of criterion `c` across responses (how often satisfied) | 0…1 | mid-range |
| `low_signal_criteria` | criteria with `coverage ≤ 0.05` or `≥ 0.95` (never vary → no signal) | list | `[]` |
| `redundant_pairs` | pairs `(a, b, corr)` whose score columns correlate `≥ 0.9` | list | `[]` |
| `weight_sensitivity[c]` | `1 − Spearman(ranking, ranking-without-c)` — how much dropping `c` changes the response ranking | 0…1 | — (higher = more influential) |

**What "low-signal" means.** A criterion is *low-signal* when its score is almost the
same for **every** response — either nearly everyone passes it (`coverage ≥ 0.95`) or nearly
everyone fails it (`coverage ≤ 0.05`). A near-constant criterion can't *distinguish* a good
answer from a bad one, so it adds no useful signal to the reward — it's dead weight in the
rubric (and it's the only thing that lowers the `structure` sub-score:
`1 − low_signal_criteria / total`). Two common causes: the criterion is trivially satisfied
(or impossible) for this prompt type, or it's worded so vaguely the grader always gives the
same score. Fix: sharpen it, split it, drop it, or add cases where it should genuinely
differ. (Distinct from `redundant_pairs`, where two criteria *vary* but always move
**together** — one of them is duplicating the other.)

### `criterion_order`
Re-grades each response with the rubric's criteria in shuffled order (always incl. the
reverse) and measures reward drift — a pointwise judge should be order-invariant
(arXiv:2602.02219).

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `mean_drift` | mean `|reward(shuffled) − reward(original)|` over responses | 0…1 | **0** |
| `max_drift` | worst single drift observed | 0…1 | **0** |
| `flip_rate` | fraction of responses whose pass/fail vector changed under reorder | 0…1 | **0** |
| `order_invariant` | `True` if `mean_drift ≤ 0.05` | bool | `True` |

### `alignment` (only with `human_score`)
Compares reward to human scores (human scores are min-max normalized to `[0,1]` first).

| Field | Meaning | Range | Good |
| --- | --- | --- | --- |
| `correlation` | `Spearman(reward, human)` — rank agreement | −1…1 | **+1** |
| `qwk` | quadratic-weighted kappa over 5 score bins (drives the sub-score) | −1…1 | **+1** |
| `kappa` | Cohen's kappa over the bins | −1…1 | **+1** |
| `calibration_error` | mean `|reward − human|` (do the *values*, not just ranks, match) | 0…1 | **0** |

> ⚠️ **Sign trap:** monotonicity's `spearman` is good at **−1**, but alignment's
> `correlation` is good at **+1**. The report hides this by normalizing both to "1 = good";
> it only matters when reading the raw JSON.

---

## Probes (used by `hacking`) & the degradation ladder (used by `monotonicity`)

All deterministic, so audits are reproducible.

| Probe | Edit applied to the answer |
| --- | --- |
| `keyword_stuff` | append the INCLUDE-criteria keywords verbatim |
| `verbosity_pad` | append content-free filler (no keywords), ×3 |
| `confident_wrong` | append a confidently-asserted, dismissive false claim |
| `format_mimic` | prepend a heading + bullet list of the keywords (structure without content) |

A good reward should give every probe variant a reward **no higher** than the original.

**`degradation_ladder`** — `rungs = 4` versions of each answer: rung 0 = original, each lower
rung truncates a growing fraction of sentences (by characters if too few sentences).

---

## Statistical & rubric concepts

- **Spearman correlation** — rank correlation in `[−1, +1]`; compares the *order* of two
  lists, ignoring their actual values. `+1` same order, `0` unrelated, `−1` opposite. Ties
  are average-ranked.
- **QWK (quadratic-weighted kappa)** — agreement on binned scores that penalizes *large*
  disagreements more than small ones; `1` = perfect, `0` = chance.
- **Cohen's kappa** — agreement on binned scores corrected for chance.
- **Bootstrap CI** — confidence interval from resampling responses (default 1000 resamples);
  reported on hacking, monotonicity, and alignment so you see the uncertainty, not just a
  point estimate. A result is "significant" when its CI excludes 0.
- **Criterion `polarity`** — `include` (the answer *should* satisfy it) or `avoid` (the answer
  should *not*). Probes only stuff `include`-criteria keywords, since tripping `avoid` ones
  would lower the reward and mask the hack.
- **Criterion `weight`** — relative importance in the weighted-mean reward aggregation.

---

See also: [interpreting-the-report.md](interpreting-the-report.md) — how to read a report card
and decide whether a reward is safe to train on.
