# Reward Report Card — healthbench_demo

**⚠️ Hackable — 'keyword_stuff' gains +0.31 reward without real quality. Trust score 0.81. Fix the rubric before training.**

- Responses audited: 10
- Composite trust score: **0.81**  (0–1, higher is better)

## Diagnostics

| Diagnostic | Score | What it means |
| --- | --- | --- |
| hacking | 0.69 | gameable via keyword_stuff — leakiest: see_doctor, warning_symptoms, drug_interactions |
| monotonicity | 0.49 | does not reliably track quality (2 inversions) |
| stability | 1.00 | identical on re-grade |
| structure | 1.00 | 1 redundant pair(s) |
| criterion_order | 1.00 | order-invariant |
| alignment | 0.70 | matches human scores |

Raw metrics (Spearman, hack gain, per-criterion gameability, CIs, QWK) are in the JSON output — run with `--out report.json`.
