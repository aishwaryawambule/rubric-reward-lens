# Verifying rubric-reward-lens on a real scenario

This guide tests the tool on **real, open-source data** (not the synthetic demo) and
**independently verifies** that its numbers are correct. Everything here runs offline,
with no API key, in a few minutes.

Three things get checked:

1. **It runs on real data** — pull real prompts + responses + human scores and audit them end-to-end.
2. **Its verdict is correct** — a naive keyword judge *should* be flagged as gameable and poorly aligned; the tool says so.
3. **Its math is trustworthy** — we recompute one diagnostic from scratch and confirm it matches the tool to 4 decimals.

The dataset is [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2):
real model responses to real prompts, each with a human **helpfulness** rating (0–4,
normalized here to a `human_score` in `[0,1]`).

---

## 0. Setup

```bash
git clone https://github.com/aishwaryawambule/rubric-reward-lens
cd rubric-reward-lens
python3 -m venv .venv
source .venv/bin/activate
pip3 install .          # NOT "-e ." — editable installs can silently fail on Python 3.14
```

> Requires Python ≥ 3.11. No GPU, no `torch`.

---

## 1. Pull real data from Hugging Face

No `datasets` install needed — this uses the HF datasets-server JSON API.

```bash
python examples/load_helpsteer2.py --n 150 --out examples/helpsteer2_responses.json
```

Expected:

```
wrote 150 real responses -> examples/helpsteer2_responses.json
```

Each entry is a real `{prompt, response, human_score}` triple.

---

## 2. Audit the real data

Audit a **naive keyword-presence judge** (`FakeGrader`) against the real responses,
using a general-quality rubric and the real human scores:

```bash
rrl audit \
  --rubric examples/helpfulness_rubric.yaml \
  --grader examples/grader.fake.yaml \
  --responses examples/helpsteer2_responses.json \
  --human-labels
```

Expected verdict (numbers will be close to this):

```
⚠️ Hackable — 'keyword_stuff' gains +0.64 reward without real quality. Trust score 0.54.
...
## Reward hacking
- Overall hack gain: 0.640 ...  Hackable: True
## Human alignment
- Correlation: 0.179; QWK: 0.076; calibration error: 0.449
```

**How to read it:** a keyword-presence judge is trivially gamed (`+0.64` from stuffing)
and barely tracks real human helpfulness (`correlation 0.179`). That is the **correct**
diagnosis of a bad judge — the tool caught it on data it has never seen.

---

## 3. Independently verify the tool's math ← the trust step

Don't take `0.179` on faith. This script recomputes the reward-vs-human rank correlation
**from scratch** (a tie-averaged Spearman written by hand) and checks it against the
tool's own `run_alignment`:

```bash
python examples/verify_alignment.py \
  --rubric examples/helpfulness_rubric.yaml \
  --responses examples/helpsteer2_responses.json
```

Expected:

```
responses               : 150 real, human-labelled
independent Spearman     : 0.1794
tool run_alignment()     : 0.1794
RESULT                   : PASS ✅ tool matches independent recompute
```

If the independent recompute matches the tool, the diagnostic is trustworthy on this
data. (Note: a *naive* Spearman that breaks ties arbitrarily gives ~0.12 and looks like
a mismatch — the keyword judge's rewards are heavily tied, so correct Spearman must
average tied ranks. That subtlety is exactly why an independent check matters.)

---

## 4. (Optional) The real production scenario — a real LLM judge

Steps 1–3 audit a *naive* judge offline. The real "is my production judge gameable?"
test swaps in an actual LLM — same data, one config change. Two ways:

### 4a. Local model via Ollama — free, no API key, no data leaves your machine

```bash
ollama serve            # if not already running
ollama pull qwen2.5:14b # any model you like; smaller = faster, larger = better judge

rrl audit \
  --rubric examples/helpfulness_rubric.yaml \
  --grader examples/grader.ollama.yaml \         # type: ollama, model: qwen2.5:14b
  --responses examples/helpsteer2_responses.json \
  --human-labels --out report.html
```

> A full audit makes many grade calls (probes × responses), so a local model can take a
> while — start with a small `--n` slice in step 1 (e.g. `--n 20`) and a small model.

### 4b. Hosted model via OpenRouter — a few cents

```bash
export OPENROUTER_API_KEY="sk-or-..."          # from openrouter.ai
rrl audit \
  --rubric examples/helpfulness_rubric.yaml \
  --grader examples/grader.openrouter.yaml \    # type: openrouter, model: anthropic/claude-haiku-4.5
  --responses examples/helpsteer2_responses.json \
  --human-labels --out report.html
```

Either way, this grades the real responses with a real model and reports whether *that*
judge is gameable and how well it aligns with the real human scores.

---

## What "it works" looks like

| Check | Pass criterion |
| --- | --- |
| Runs on real data | Step 1 writes 150 responses; step 2 prints a verdict, no crash |
| Verdict is correct | Naive keyword judge → `Hackable: True`, low alignment correlation |
| Math is trustworthy | Step 3 prints `PASS ✅` (independent recompute matches the tool) |
| Full unit suite | `pytest` → all tests pass |

Run the suite too:

```bash
pip3 install ".[dev]"
pytest
```
