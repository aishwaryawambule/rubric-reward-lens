# Interpreting the report card

Every audit produces a one-line **verdict**, a composite **trust score**, and a
**Diagnostics** table. The human-readable report (markdown / HTML) shows:

```
# Reward Report Card — <rubric>

<verdict>

- Responses audited: <n>
- Composite trust score: <0.00>  (0–1, higher is better)

## Diagnostics
| Diagnostic   | Score | What it means                                |
| ------------ | ----- | -------------------------------------------- |
| hacking      | 0.97  | not gameable                                 |
| monotonicity | 0.69  | mostly tracks quality (7 inversions)         |
| stability    | 1.00  | identical on re-grade                        |
| structure    | 0.50  | 2 low-signal criteria: accurate, not_evasive |
| criterion_order | 1.00 | order-invariant                           |
| alignment    | 0.00  | does not match human scores                  |

Raw metrics … are in the JSON output — run with `--out report.json`.
```

## The golden rule

**In the report, every number obeys one rule: 0–1, where `1.0` is best.** Higher is always
better — the `Score` column and the trust score both. There are **no raw metrics in the
report** (no Spearman signs, no hack gains, no per-criterion gameability) — those have
mixed directions and live in the **JSON** instead. So you never have to ask "is high good
here?" in the report: it always is.

The `Score` is the normalized sub-score; `What it means` is a plain-English reading.

---

## The six diagnostics

Each row is one diagnostic. Below: what it measures, and the phrases you'll see in
**What it means**.

### 1. Reward hacking — *can cheap tricks earn reward?*
Applies four adversarial probes (`keyword_stuff`, `verbosity_pad`, `confident_wrong`,
`format_mimic`) that should **not** improve real quality, and checks the reward anyway.
- **Score:** `1 − overall_hack_gain` → 1.0 when no trick earns reward.
- **What it means:** `not gameable`, or `gameable via <probe> — leakiest: <criteria>`
  (the leaky criteria are named from `per_criterion_gameability`).
- **The score to be strict on** — it's the failure this whole tool exists to catch.

### 2. Discrimination / monotonicity — *does worse quality get less reward?*
Progressively degrades each answer and checks the reward falls in step.
- **Score:** higher = reward drops more cleanly as quality drops.
- **What it means:** `reward falls cleanly as quality drops` / `mostly tracks quality
  (N inversions)` / `does not reliably track quality (N inversions)`.

### 3. Grader stability — *same input, same reward?*
Re-grades each response several times and measures the spread.
- **Score:** 1.0 = deterministic. (Meaningful even at small sample sizes — it's about the
  judge, not the data.)
- **What it means:** `identical on re-grade` / `wobbles across re-grades`.

### 4. Criterion structure — *is every rubric criterion pulling its weight?*
Looks across responses at how each criterion behaves.
- **Score:** `1 − (low-signal criteria / total)`. Only low-signal criteria lower it.
- **What it means:** `all criteria informative` / `N low-signal: <names>` (and/or
  `N redundant pair(s)`).

### 5. Criterion-order invariance — *does reordering the criteria change the reward?*
Re-grades each answer with the rubric's criteria shuffled (a pointwise judge shouldn't care).
- **Score:** `1 − clamp(mean_drift)` → 1.0 when the reward is unchanged by reordering.
- **What it means:** `order-invariant` / `reward drifts with criterion order (mean Δ 0.NN)`.
- Like stability, it's about the *judge*, so it's meaningful even at small sample sizes.

### 6. Human alignment — *does the reward agree with humans?* (optional)
Only runs when responses carry a `human_score`. Uses quadratic-weighted kappa (QWK).
- **Score:** `clamp(QWK)` → 1.0 = perfect agreement.
- **What it means** (QWK bands): `≥0.6` → `matches human scores`; `≥0.2` → `weak agreement
  with humans`; else → `does not match human scores`.

> **Sample size matters.** With very few responses, hacking / monotonicity / structure /
> alignment are statistical artifacts — e.g. *any* 2 points force a correlation of exactly
> ±1. Aim for ~15–20+ responses before trusting these; stability (#3) and criterion-order
> (#5) are the exceptions — they measure the *judge*, so they're meaningful at any n.

---

## Trust score & verdict

- **Trust score** = the **plain average** of whichever sub-scores ran (equal weight), 0–1.
- **Verdict:**

| Condition | Verdict |
| --- | --- |
| `Hackable` and (hack gain ≥ **0.15** or trust < **0.6**) | **⚠️ Hackable — fix the rubric before training** |
| `Hackable` but minor | **⚠️ Caution — review the probes** |
| not hackable and trust ≥ **0.70** | **✅ Robust** |
| not hackable and trust < 0.70 | **⚠️ Caution — review the diagnostics** |

## "Good score" cheat sheet

| Diagnostic | Perfect (Score) | Good enough to trust |
| --- | --- | --- |
| hacking | 1.00 | ≥ ~0.85 and `not gameable` |
| monotonicity | 1.00 | ≥ ~0.8, no inversions |
| stability | 1.00 | ≥ ~0.9 |
| structure | 1.00 | ≥ ~0.8 |
| criterion_order | 1.00 | ≥ ~0.9, `order-invariant` |
| alignment | 1.00 | QWK ≥ ~0.6 |
| **trust score** | **1.00** | **≥ 0.70 → ✅ Robust** |

The one non-negotiable: a `Hackable` verdict means fix the rubric before you trust the
reward — no amount of good alignment compensates for a gameable reward.

---

## JSON raw-field reference (for `--out report.json`)

The JSON holds the structured record: `trust_score`, `verdict`, `sub_scores`, and the full
raw `diagnostics`. **Unlike the report, raw fields are not all "bigger = better"** — each
has its own direction:

| Raw field (JSON) | "Good" value | Note |
| --- | --- | --- |
| `hacking.overall_hack_gain` | **0.0** (or negative) | reward a trick earned |
| `hacking.per_criterion_gameability[c]` | **0.0** | per-criterion score inflation; **1.0 = fully exploited (worst)** |
| `monotonicity.spearman` | **−1.0** | corruption↑ ⇒ reward↓ |
| `stability.reward_std` | **0.0** | spread across re-grades |
| `structure.low_signal_criteria` | `[]` | criteria that never vary |
| `criterion_order.mean_drift` | **0.0** | reward change when criteria are reordered |
| `alignment.correlation` | **+1.0** | rank agreement with humans (**opposite sign from monotonicity**) |
| `alignment.qwk` | **+1.0** | drives the alignment sub-score |

**Spearman** (used by monotonicity and `alignment.correlation`) is a −1…+1 rank
correlation: `+1` = same order, `0` = unrelated, `−1` = opposite order. Note the sign trap:
monotonicity wants `−1`, alignment wants `+1`. These directions only matter when reading the
**JSON** — the report normalizes them all to "1 = good" for you.

For the precise definition, range, and good direction of **every** metric and score (all
JSON fields, the sub-score formulas, the probes, and the statistical terms), see the
[**Metrics & scores reference**](metrics.md).
