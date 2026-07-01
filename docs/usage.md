# Using rubric-reward-lens

There are **two ways** to use the tool — as a **Python library** (embed it in your own
code) or from the **command line**. Same engine underneath; the CLI is a thin wrapper that
loads your files and calls `audit()`.

| Use the… | When you want to… |
| --- | --- |
| **Library** (`audit(...)`) | embed in an eval pipeline / CI gate / RL setup, read scores programmatically, plug in a custom grader |
| **CLI** (`rrl audit`) | run a quick one-off audit from the shell, generate a shareable HTML/JSON report, script it in CI |

---

## 1. As a Python library (code integration)

Import the public API, call `audit()`, and inspect the returned `ReportCard` in code:

```python
from rubric_reward_lens import audit, Rubric, Response, OllamaGrader

rubric = Rubric.from_yaml("my_rubric.yaml")            # or build it in code (see below)
grader = OllamaGrader(model="llama3.1:latest")          # or OpenRouterGrader / FakeGrader / your own
responses = [
    Response(id="r1", prompt="...", text="...", human_score=0.8),
    Response(id="r2", prompt="...", text="..."),        # human_score optional
]

card = audit(rubric, grader, responses, human_labels=True)

card.verdict          # "⚠️ Caution — trust score 0.65 ..."
card.trust_score      # 0.65
card.sub_scores()     # {"hacking": 1.0, "criterion_order": 0.82, "alignment": 0.0, ...}
card.to_dict()        # full structured record (all raw fields)
card.to_json("report.json")   # or .to_html(...) / .to_markdown()
```

Building inputs in code instead of files:

```python
from rubric_reward_lens import Rubric, Criterion

rubric = Rubric(criteria=[
    Criterion("addresses_q", "directly answers the question", weight=2.0),           # polarity defaults to include
    Criterion("no_hedging",  "avoids vague non-answers", weight=1.0, polarity="avoid"),
], name="my-rubric")
```

**Use your own grader** — anything with `grade(rubric, response) -> GraderResult` works:

```python
from rubric_reward_lens import CriterionScore, GraderResult, aggregate

class MyGrader:
    def grade(self, rubric, response):
        scores = [CriterionScore(c.id, my_judge(c, response.text)) for c in rubric.criteria]
        return GraderResult(response.id, scores, aggregate(rubric, scores))

card = audit(rubric, MyGrader(), responses)
```

**Add your own hack probe** — pass `probes=` to `audit()` to test domain-specific gaming
(sycophancy, prompt-injection, fake citations…). A probe is a deterministic
`(response, rubric) -> response` transform that a good reward should *not* pay for:

```python
from rubric_reward_lens.probes import Probe
from rubric_reward_lens.models import Response

def sycophancy(response, rubric):
    flattery = " You are absolutely right, this is an excellent and correct question."
    return Response(id=f"{response.id}::syco", text=response.text + flattery,
                    prompt=response.prompt, human_score=response.human_score)

card = audit(rubric, grader, responses, probes=[Probe("sycophancy", "down", sycophancy)])
```

`probes=None` (the default) uses the four built-ins (`keyword_stuff`, `verbosity_pad`,
`confident_wrong`, `format_mimic`); passing a list replaces them.

**Add your own diagnostic** — pass `extra_diagnostics=` to `audit()`. A diagnostic is a
callable `(rubric, grader, responses) -> DiagnosticResult`; its 0–1 `score` (1 = good) flows
into the trust score, and it appears in the report table and JSON alongside the built-ins:

```python
from rubric_reward_lens import DiagnosticResult

def brevity(rubric, grader, responses):
    avg = sum(len(r.text) for r in responses) / len(responses)
    return DiagnosticResult(
        name="brevity", score=1.0 if avg < 400 else 0.5,
        summary=f"avg answer {avg:.0f} chars", raw={"avg_chars": avg},
    )

card = audit(rubric, grader, responses, extra_diagnostics=[brevity])
```

> **Code-only:** custom `probes=` / `extra_diagnostics=` work through the library, **not**
> the CLI — a YAML config can't hold a Python function. If you drive the tool from a shell
> and need a custom probe/diagnostic, write a short script that calls `audit(...)`.

Public API: `audit`, `Rubric`, `Criterion`, `Response`, `CriterionScore`, `GraderResult`,
`DiagnosticResult`, `Probe`, `aggregate`, `FakeGrader`, `OllamaGrader`, `OpenRouterGrader`,
`ReportCard`, `load_demo`.

Typical integration: gate a build or an RL run on the result, e.g.

```python
card = audit(rubric, grader, responses)
if card.hacking and card.hacking.hackable:
    raise SystemExit(f"reward is gameable — {card.verdict}")   # block training / fail CI
```

---

## 2. From the CLI

Installed as the `rrl` command:

```bash
rrl demo                                    # offline smoke test on the bundled synthetic data

rrl audit \
  --rubric my_rubric.yaml \                 # yaml or json
  --grader grader.yaml \                    # {type: fake | ollama | openrouter, ...}
  --responses responses.json \              # [{"id","prompt","text","human_score"?}, ...]
  --human-labels \                          # enable the alignment diagnostic (needs human_score)
  --out report.html                         # .html / .json / .md — omit to print markdown to stdout
```

Grader config examples:

```yaml
# grader.yaml — pick one
type: fake                                  # offline keyword grader, no LLM (for demos/tests)
# type: ollama
# model: llama3.1:latest                     # a local model (needs `ollama serve` + the model pulled)
# type: openrouter
# model: anthropic/claude-haiku-4.5           # a hosted model (needs OPENROUTER_API_KEY)
```

The verdict is printed to the terminal; the full report goes to `--out`. Prefer
`--out report.json` when you want the durable, complete record (all raw metrics).

---

## Which output format?

- **markdown / HTML** — human-readable: verdict, trust score, and the per-diagnostic table
  (every score 0–1, 1 = good, plus a plain-English reading).
- **JSON** — the complete structured record, including all raw metric fields. Use it to
  re-render or analyze without re-running the grader.

To interpret the result, see [interpreting-the-report.md](interpreting-the-report.md); for
the precise definition of every metric and score, see [metrics.md](metrics.md).
