# rubric-reward-lens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `pip`-installable Python library + CLI that audits a rubric-based LLM reward signal for gameability and quality *before* it is used for RL.

**Architecture:** Pure-Python, inference-only. A `Grader` interface (LLM-backed in production, deterministic `FakeGrader` in tests) turns `(rubric, response)` into per-criterion scores + a scalar reward. A set of independent **diagnostics** (label-free: hacking probes, monotonicity, stability, criterion-structure; optional: human-alignment) run over a response set and produce a `ReportCard` with a composite trust verdict, rendered to JSON/markdown/HTML.

**Tech Stack:** Python 3.11+, numpy, scipy, pyyaml, httpx. Test: pytest. No GPU, no torch, no Jinja2 (stdlib templating).

## Global Constraints

- Python 3.11+ (`requires-python = ">=3.11"`). Dev env is 3.14.
- Runtime deps: `numpy`, `scipy`, `pyyaml`, `httpx`. No other runtime deps.
- Package import name: `rubric_reward_lens`. Distribution name: `rubric-reward-lens`. CLI: `rrl`.
- `src/` layout. Tests import via `pythonpath = ["src"]` (configured in `pyproject.toml`); no editable install required to run tests.
- Inference-only: NO training, NO torch, NO learned reward models.
- All randomness seeded (default `seed=0`). No `Date.now()`-style nondeterminism in core logic.
- Every headline metric carries a bootstrap confidence interval.
- Tests must run fully offline: no network, no API key. Network code (`OpenRouterGrader`) is tested with an injected fake transport.
- License: MIT. Demo data is **synthetic** (clearly labeled), not real HealthBench.
- Run tests with: `cd /Users/aishwaryarai/Documents/E2/rubric-reward-lens && python3 -m pytest -q`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | packaging, deps, pytest config (`pythonpath=src`) |
| `LICENSE` | MIT |
| `README.md` | what it does, install, quickstart, demo |
| `.gitignore` | venv, caches, `*.html` outputs |
| `src/rubric_reward_lens/__init__.py` | public API re-exports |
| `src/rubric_reward_lens/models.py` | `Polarity, Criterion, Rubric, Response, CriterionScore, GraderResult` |
| `src/rubric_reward_lens/grader.py` | `Grader` protocol, `aggregate()`, `FakeGrader` |
| `src/rubric_reward_lens/openrouter.py` | `OpenRouterGrader` (httpx, injectable transport) + JSON parse |
| `src/rubric_reward_lens/stats.py` | bootstrap CI, spearman, kappa/qwk, paired diff, significance |
| `src/rubric_reward_lens/probes.py` | hack-probe transforms + degradation ladder |
| `src/rubric_reward_lens/diagnostics/__init__.py` | re-exports |
| `src/rubric_reward_lens/diagnostics/hacking.py` | A1 reward-hacking probes |
| `src/rubric_reward_lens/diagnostics/monotonicity.py` | A2 discrimination/monotonicity |
| `src/rubric_reward_lens/diagnostics/stability.py` | A3 grader stability |
| `src/rubric_reward_lens/diagnostics/structure.py` | B1-3 criterion structure |
| `src/rubric_reward_lens/diagnostics/alignment.py` | C1-2 human alignment (optional) |
| `src/rubric_reward_lens/report.py` | `ReportCard`, composite trust score, renderers |
| `src/rubric_reward_lens/audit.py` | `audit()` orchestrator |
| `src/rubric_reward_lens/cli.py` | `rrl audit`, `rrl demo` |
| `src/rubric_reward_lens/data/__init__.py` | `load_demo()` + synthetic sample loader |
| `src/rubric_reward_lens/data/healthbench_demo/*.json` | synthetic demo rubric + responses + human scores |
| `tests/conftest.py` | shared fixtures: `KeywordGrader`, `RobustGrader`, `NoisyGrader`, sample rubric/responses |
| `tests/test_*.py` | one per module |

---

## Interfaces (locked — every task must match these exactly)

```python
# models.py
class Polarity(str, Enum): INCLUDE = "include"; AVOID = "avoid"

@dataclass(frozen=True)
class Criterion:
    id: str; text: str; weight: float = 1.0; polarity: Polarity = Polarity.INCLUDE

@dataclass
class Rubric:
    criteria: list[Criterion]; name: str = ""
    @classmethod
    def from_dict(cls, d: dict) -> "Rubric"
    @classmethod
    def from_yaml(cls, path: str) -> "Rubric"
    @classmethod
    def from_json(cls, path: str) -> "Rubric"
    def criterion_ids(self) -> list[str]
    def get(self, cid: str) -> Criterion

@dataclass
class Response:
    id: str; text: str; prompt_id: str = ""; prompt: str = ""; human_score: float | None = None

@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str; score: float; justification: str = ""   # score in [0,1]

@dataclass
class GraderResult:
    response_id: str; per_criterion: list[CriterionScore]; reward: float
    def score_for(self, cid: str) -> float

# grader.py
class Grader(Protocol):
    def grade(self, rubric: Rubric, response: Response) -> GraderResult: ...

def aggregate(rubric: Rubric, per_criterion: list[CriterionScore]) -> float
    # weighted mean of scores by criterion weight, in [0,1]; empty -> 0.0

class FakeGrader:   # deterministic, keyword-presence based (intentionally hackable)
    def __init__(self, keywords: dict[str, list[str]] | None = None, window: int | None = None)
    def grade(self, rubric, response) -> GraderResult
    # criterion satisfied (1.0) if any of its keywords appears in response.text
    # (within first `window` chars if set), else 0.0; AVOID polarity inverts.
    # keywords default: content words (lowercased, len>3, minus stopwords) from criterion.text

# openrouter.py
class OpenRouterGrader:
    def __init__(self, model: str, prompt_template: str | None = None,
                 temperature: float = 0.0, api_key: str | None = None, transport=None)
    def grade(self, rubric, response) -> GraderResult
def parse_grader_json(raw: str, rubric: Rubric, response_id: str) -> GraderResult

# stats.py
def bootstrap_ci(values, statistic=np.mean, n_boot=1000, ci=0.95, seed=0) -> tuple[float,float,float]
    # returns (point_estimate, ci_low, ci_high)
def paired_bootstrap_diff(a, b, n_boot=1000, ci=0.95, seed=0) -> tuple[float,float,float]
def spearman(x, y) -> float
def cohen_kappa(a, b) -> float
def quadratic_weighted_kappa(a, b, n_bands: int) -> float
def significant(ci_low: float, ci_high: float) -> bool   # True if interval excludes 0

# probes.py
@dataclass(frozen=True)
class Probe:
    name: str; expected_direction: str  # "down" = variant reward should be <= original
    def apply(self, response: Response, rubric: Rubric) -> Response

def keyword_stuff(response, rubric) -> Response
def verbosity_pad(response, rubric) -> Response
def confident_wrong(response, rubric) -> Response
def format_mimic(response, rubric) -> Response
def degradation_ladder(response, rungs: int = 4) -> list[Response]   # rung 0 = original, worse downward
HACK_PROBES: list[Probe]   # the four above

# diagnostics/hacking.py
@dataclass
class HackingResult:
    overall_hack_gain: float; ci: tuple[float,float]
    per_probe: dict[str, tuple[float, float, float]]          # name -> (gain, low, high)
    per_criterion_gameability: dict[str, float]
    hackable: bool
def run_hacking(rubric, grader, responses, probes=None, n_boot=1000, seed=0) -> HackingResult

# diagnostics/monotonicity.py
@dataclass
class MonotonicityResult:
    spearman: float; ci: tuple[float,float]; inversions: int; separation: float; monotonic: bool
def run_monotonicity(rubric, grader, responses, rungs=4, n_boot=1000, seed=0) -> MonotonicityResult

# diagnostics/stability.py
@dataclass
class StabilityResult:
    reward_std: float; mean_ci_width: float
    per_criterion_flip_rate: dict[str, float]; stable: bool
def run_stability(rubric, grader, responses, n_repeats=5, seed=0) -> StabilityResult

# diagnostics/structure.py
@dataclass
class StructureResult:
    redundant_pairs: list[tuple[str, str, float]]            # (c1, c2, corr)
    weight_sensitivity: dict[str, float]
    coverage: dict[str, float]                               # pass-rate per criterion
    low_signal_criteria: list[str]
def run_structure(rubric, grader, responses, redundancy_thresh=0.9, seed=0) -> StructureResult

# diagnostics/alignment.py
@dataclass
class AlignmentResult:
    correlation: float; qwk: float; kappa: float
    calibration_error: float; ci: tuple[float,float]; n: int
def run_alignment(rubric, grader, responses, n_bands=5, n_boot=1000, seed=0) -> AlignmentResult
    # uses only responses with human_score is not None

# report.py
@dataclass
class ReportCard:
    rubric_name: str; n_responses: int
    hacking: HackingResult | None = None
    monotonicity: MonotonicityResult | None = None
    stability: StabilityResult | None = None
    structure: StructureResult | None = None
    alignment: AlignmentResult | None = None
    @property
    def trust_score(self) -> float       # composite in [0,1]
    @property
    def verdict(self) -> str
    def to_dict(self) -> dict
    def to_json(self, path: str | None = None) -> str
    def to_markdown(self, path: str | None = None) -> str
    def to_html(self, path: str | None = None) -> str

# audit.py
DEFAULT_DIAGNOSTICS = ("hacking", "monotonicity", "stability", "structure")
def audit(rubric, grader, responses, diagnostics=DEFAULT_DIAGNOSTICS,
          human_labels: bool = False, n_boot=1000, seed=0) -> ReportCard

# data/__init__.py
def load_demo() -> tuple[Rubric, list[Response]]    # synthetic, includes human_score
```

**Verdict / boolean thresholds (locked):**
- `hackable` = `significant(ci_low, ci_high)` AND `overall_hack_gain > 0.05`.
- `monotonic` = `spearman <= -0.5` (reward falls as corruption rises) AND `inversions` small.
- `stable` = `reward_std <= 0.1` AND every criterion flip-rate `<= 0.2`.
- `low_signal_criteria` = pass-rate `<= 0.05` or `>= 0.95`.
- `trust_score` = mean of available sub-scores, each in [0,1]:
  hacking → `1 - clamp(overall_hack_gain,0,1)`; monotonicity → `clamp(-spearman,0,1)`;
  stability → `1 - clamp(reward_std/0.5,0,1)`; structure → `1 - frac(low_signal_criteria)`;
  alignment → `clamp(qwk,0,1)`.

---

## Tasks

### Task 1: Repo scaffold + packaging
**Files:** Create `pyproject.toml`, `LICENSE`, `.gitignore`, `src/rubric_reward_lens/__init__.py` (empty for now), `tests/conftest.py` (empty for now).
**Interfaces:** Produces a runnable pytest env.
- [ ] **Step 1** Write `tests/test_smoke.py` with `def test_import(): import rubric_reward_lens` (will fail: package empty is fine, import should succeed once `__init__.py` exists).
- [ ] **Step 2** Run `python3 -m pytest tests/test_smoke.py -q` → expect collection/import to work once files exist.
- [ ] **Step 3** Write `pyproject.toml`: `[project]` name `rubric-reward-lens`, version `0.1.0`, `requires-python=">=3.11"`, deps `numpy,scipy,pyyaml,httpx`; `[project.scripts] rrl = "rubric_reward_lens.cli:main"`; `[tool.pytest.ini_options] pythonpath=["src"]`, `testpaths=["tests"]`. Use `hatchling` build backend with `[tool.hatch.build.targets.wheel] packages=["src/rubric_reward_lens"]`. Write MIT `LICENSE` (copyright "rubric-reward-lens contributors"). Write `.gitignore`.
- [ ] **Step 4** Run `python3 -m pytest -q` → PASS.
- [ ] **Step 5** Commit: `chore: scaffold package + pytest config`.

### Task 2: Data models (`models.py`)
**Files:** Create `src/rubric_reward_lens/models.py`, `tests/test_models.py`.
**Interfaces:** Produces `Polarity, Criterion, Rubric, Response, CriterionScore, GraderResult` per the locked interface.
- [ ] **Step 1** Tests: round-trip `Rubric.from_dict({"name":"r","criteria":[{"id":"c1","text":"mentions a doctor","weight":2}]})` → 1 criterion, weight 2.0, polarity INCLUDE; `criterion_ids()==["c1"]`; `get("c1").text`; `GraderResult(... ).score_for("c1")` returns the score; `from_json`/`from_yaml` load a tmp file. Include a test that unknown `get("nope")` raises `KeyError`.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement dataclasses + classmethods. `from_yaml` uses `yaml.safe_load`; `from_json` uses `json.load`; both delegate to `from_dict`. `from_dict` coerces `weight` to float, `polarity` to `Polarity`.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: core data models`.

### Task 3: Grader interface + aggregation + FakeGrader (`grader.py`)
**Files:** Create `src/rubric_reward_lens/grader.py`, `tests/test_grader.py`.
**Interfaces:** Consumes `models`. Produces `Grader`, `aggregate`, `FakeGrader`.
- [ ] **Step 1** Tests: `aggregate` weighted mean (criteria weights 1 and 3, scores 0 and 1 → 0.75); empty → 0.0. `FakeGrader().grade(rubric, response)` returns `GraderResult` with reward in [0,1]; a response containing the criterion's content word scores 1.0 on it, one without scores 0.0; AVOID polarity inverts; `window=50` means a keyword after char 50 does NOT count.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement. Keyword extraction: lowercase criterion text, split on non-alpha, drop a small stopword set + tokens len≤3, dedupe. `FakeGrader.grade` builds `CriterionScore` per criterion then `aggregate`.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: grader interface, aggregation, FakeGrader`.

### Task 4: OpenRouter grader (`openrouter.py`)
**Files:** Create `src/rubric_reward_lens/openrouter.py`, `tests/test_openrouter.py`.
**Interfaces:** Consumes `models, grader.aggregate`. Produces `OpenRouterGrader`, `parse_grader_json`.
- [ ] **Step 1** Tests (NO network): `parse_grader_json('{"per_criterion":[{"criterion_id":"c1","score":1,"justification":"ok"}]}', rubric, "r1")` → GraderResult with reward computed via aggregate; tolerates code-fenced JSON and extra prose (extract first `{`..last `}`); missing criterion → score 0.0; out-of-range score clamped to [0,1]. `OpenRouterGrader(..., transport=fake)` where `fake` is an `httpx.MockTransport` returning a canned chat-completion JSON → `.grade()` returns parsed result. Construct client as `httpx.Client(transport=transport)`.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement. Build OpenAI-compatible chat request to `https://openrouter.ai/api/v1/chat/completions`; system+user prompt embeds rubric criteria and asks for strict JSON `{per_criterion:[{criterion_id,score,justification}]}` with score in [0,1]; `api_key` from arg or `OPENROUTER_API_KEY`; allow injected `transport`. Parse `choices[0].message.content` via `parse_grader_json`.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: OpenRouter grader with injectable transport`.

### Task 5: Stats utilities (`stats.py`)
**Files:** Create `src/rubric_reward_lens/stats.py`, `tests/test_stats.py`.
**Interfaces:** Produces bootstrap_ci, paired_bootstrap_diff, spearman, cohen_kappa, quadratic_weighted_kappa, significant.
- [ ] **Step 1** Tests: `bootstrap_ci([0.5]*100)` → point≈0.5, low≤0.5≤high, deterministic across two calls (same seed). `paired_bootstrap_diff(a,b)` where a>b elementwise → positive point, `significant(low,high)` True. `spearman([1,2,3],[1,2,3])==1.0`, `==-1.0` reversed. `cohen_kappa` perfect agreement →1.0. `quadratic_weighted_kappa([0,1,2],[0,1,2],3)==1.0`. `significant(0.1,0.3)` True, `significant(-0.1,0.2)` False.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement with numpy RNG seeded (`np.random.default_rng(seed)`), `scipy.stats.spearmanr`, `scipy.stats.cohen_kappa_score`? (use sklearn-free: implement kappa with numpy). Implement QWK with the standard confusion-matrix formula. Percentile bootstrap.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: statistics (bootstrap CIs, spearman, kappa, qwk)`.

### Task 6: Probes (`probes.py`)
**Files:** Create `src/rubric_reward_lens/probes.py`, `tests/test_probes.py`.
**Interfaces:** Consumes `models, grader` (keyword extraction reused/duplicated minimally). Produces probe functions, `Probe`, `HACK_PROBES`, `degradation_ladder`.
- [ ] **Step 1** Tests: `keyword_stuff(resp, rubric).text` contains every criterion's keywords and is longer than original; `verbosity_pad` longer, same information core; `confident_wrong` adds a flagged false-claim sentence; `format_mimic` adds markdown headers/bullets. `degradation_ladder(resp, rungs=4)` returns 4 Responses, rung 0 == original text, each subsequent strictly shorter/more corrupted; all have unique ids. Each `Probe.apply` returns a `Response` with a derived id (e.g. `f"{resp.id}::keyword_stuff"`).
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement deterministic string transforms. `keyword_stuff`: append `" "+ " ".join(all_keywords)`. `verbosity_pad`: append filler sentences ×N. `confident_wrong`: append `"It is definitely certain that <topic> is false."` per first criterion. `format_mimic`: prepend `"## Summary\n- "` bullets of first sentences. `degradation_ladder`: split into sentences, drop a growing fraction from the end per rung.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: hack probes + degradation ladder`.

### Task 7: Hacking diagnostic (`diagnostics/hacking.py`)
**Files:** Create `src/rubric_reward_lens/diagnostics/hacking.py`, `tests/test_hacking.py`.
**Interfaces:** Consumes `models, grader, probes, stats`. Produces `HackingResult`, `run_hacking`.
- [ ] **Step 1** Tests: with `KeywordGrader` (fixture = `FakeGrader`) and a rubric, `run_hacking` over responses returns `hackable=True` and `overall_hack_gain>0.05` (keyword_stuff raises reward). With `RobustGrader` (fixture, `FakeGrader(window=40)` so appended stuffing after char 40 doesn't help) → `hackable=False`. `per_probe` has an entry per probe; `per_criterion_gameability` keys ⊆ criterion ids. CI present.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: for each response, grade original; for each hack probe, grade variant; gain = reward(variant)-reward(original). Pool gains across responses → `bootstrap_ci`. Per-probe CIs. Per-criterion gameability = mean positive criterion-score increase under any probe. `hackable` per locked threshold.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: reward-hacking diagnostic`.

### Task 8: Monotonicity diagnostic (`diagnostics/monotonicity.py`)
**Files:** Create file + `tests/test_monotonicity.py`.
**Interfaces:** Consumes `models, grader, probes.degradation_ladder, stats`. Produces `MonotonicityResult`, `run_monotonicity`.
- [ ] **Step 1** Tests: with a `LengthGrader` fixture (reward ∝ response length / content retained) and a degradation ladder, `spearman` strongly negative, `monotonic=True`, `inversions` low, `separation>0`. With a `ConstantGrader` (reward always 0.5) → `spearman≈0`, `monotonic=False`.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: per response build ladder, grade each rung, compute spearman(rung_index, reward) (expect negative), average across responses; pool for CI; count inversions (adjacent rung where reward rises); separation = mean(reward[0]-reward[-1]).
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: monotonicity/discrimination diagnostic`.

### Task 9: Stability diagnostic (`diagnostics/stability.py`)
**Files:** Create file + `tests/test_stability.py`.
**Interfaces:** Consumes `models, grader, stats`. Produces `StabilityResult`, `run_stability`.
- [ ] **Step 1** Tests: with `FakeGrader` (deterministic) → `reward_std==0`, all flip-rates 0, `stable=True`. With `NoisyGrader` fixture (randomizes per-criterion pass with prob 0.5, seeded by call count) → `reward_std>0`, some flip-rate>0, `stable=False`.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: grade each response `n_repeats` times; reward_std = mean over responses of std of rewards; per-criterion flip_rate = mean over responses of fraction of repeats differing from modal pass/fail; mean_ci_width from bootstrap of rewards.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: grader-stability diagnostic`.

### Task 10: Structure diagnostic (`diagnostics/structure.py`)
**Files:** Create file + `tests/test_structure.py`.
**Interfaces:** Consumes `models, grader`. Produces `StructureResult`, `run_structure`.
- [ ] **Step 1** Tests: build responses where criteria c1,c2 always co-fire (correlation 1.0) and c3 always passes → `redundant_pairs` includes (c1,c2); `low_signal_criteria` includes c3; `coverage` pass-rates correct; `weight_sensitivity` returns a value per criterion.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: grade all responses → matrix [responses × criteria] of scores. Coverage = column means. Redundancy = pairwise Pearson corr (numpy) ≥ thresh (guard zero-variance columns). Weight-sensitivity = Spearman distance between reward ranking with all criteria vs. with criterion i dropped. low_signal per threshold.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: criterion-structure diagnostic`.

### Task 11: Alignment diagnostic (`diagnostics/alignment.py`)
**Files:** Create file + `tests/test_alignment.py`.
**Interfaces:** Consumes `models, grader, stats`. Produces `AlignmentResult`, `run_alignment`.
- [ ] **Step 1** Tests: responses carry `human_score`; with a grader whose reward equals (a binned function of) human_score → `correlation≈1`, `qwk` high, low calibration_error. Responses without human_score are ignored; if none have labels, raise `ValueError`.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: filter labeled responses; grade; correlation = spearman(reward, human_score); bin both into n_bands → qwk + cohen_kappa; calibration_error = mean abs gap between reward and human in matched bins; bootstrap CI on correlation.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: human-alignment diagnostic`.

### Task 12: Report card + renderers (`report.py`)
**Files:** Create `src/rubric_reward_lens/report.py`, `tests/test_report.py`.
**Interfaces:** Consumes all `*Result` types + `models`. Produces `ReportCard`.
- [ ] **Step 1** Tests: `ReportCard` with a hackable `HackingResult` → `trust_score<0.5`, `verdict` mentions "hackable"; with robust results → `trust_score>0.7`, verdict positive. `to_json` round-trips via `json.loads`; `to_markdown` contains section headers; `to_html` returns `<html` and embeds the verdict; `to_*` with a path writes the file.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement composite `trust_score` (mean of available sub-scores per locked formula), `verdict` string, `to_dict`, and three renderers using stdlib (`json`, f-strings/`string.Template` for md+html). No Jinja2.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: report card + json/markdown/html renderers`.

### Task 13: Audit orchestrator (`audit.py`)
**Files:** Create `src/rubric_reward_lens/audit.py`, `tests/test_audit.py`. Update `src/rubric_reward_lens/__init__.py` to re-export public API.
**Interfaces:** Consumes everything. Produces `audit`, `DEFAULT_DIAGNOSTICS`.
- [ ] **Step 1** Tests: `audit(rubric, FakeGrader(), responses)` returns a `ReportCard` with hacking/monotonicity/stability/structure populated, alignment None. `audit(..., diagnostics=("hacking",))` only runs hacking. `audit(..., human_labels=True)` runs alignment when labels present. `import rubric_reward_lens as rrl; rrl.audit, rrl.Rubric, rrl.FakeGrader` exist.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement: dispatch each requested diagnostic, collect into `ReportCard`. Re-export in `__init__`: `audit, Rubric, Criterion, Response, Polarity, FakeGrader, OpenRouterGrader, ReportCard, load_demo`.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: audit orchestrator + public API`.

### Task 14: Synthetic demo data (`data/`)
**Files:** Create `src/rubric_reward_lens/data/__init__.py`, `src/rubric_reward_lens/data/healthbench_demo/rubric.json`, `.../responses.json`, `tests/test_data.py`.
**Interfaces:** Produces `load_demo() -> (Rubric, list[Response])`.
- [ ] **Step 1** Tests: `load_demo()` returns a `Rubric` with ≥4 criteria and ≥8 `Response`s; ≥4 responses carry `human_score`; responses include both strong and weak answers. Data files load as valid JSON.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Author a small SYNTHETIC medical-Q&A dataset (clearly labeled synthetic; a header field `"_note":"synthetic illustrative data, not real HealthBench"`). 1-2 prompts, a weighted rubric (include + avoid criteria), ~10 responses spanning quality with human_scores. `load_demo` reads via `importlib.resources`.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: synthetic demo dataset + loader`.

### Task 15: CLI (`cli.py`)
**Files:** Create `src/rubric_reward_lens/cli.py`, `tests/test_cli.py`.
**Interfaces:** Consumes `audit, data, models, grader, openrouter`. Produces `main(argv=None)`.
- [ ] **Step 1** Tests (use `FakeGrader` via `--grader` config `{"type":"fake"}`; invoke `main([...])` in-process): `rrl demo --out <tmp>.html` writes a file and returns 0, file contains a verdict. `rrl audit --rubric <demo rubric> --grader <fake cfg> --responses <demo responses> --out <tmp>.json` writes JSON. Unknown command → nonzero exit.
- [ ] **Step 2** Run → FAIL.
- [ ] **Step 3** Implement argparse with subcommands `audit` and `demo`. Grader factory from YAML/JSON config: `type: fake|openrouter`. Output format inferred from `--out` extension (.json/.md/.html), default markdown to stdout. `main` returns int exit code.
- [ ] **Step 4** Run → PASS.
- [ ] **Step 5** Commit: `feat: rrl CLI (audit, demo)`.

### Task 16: Falsification / validation test (the proof)
**Files:** Create `tests/test_validation.py`.
**Interfaces:** Consumes `audit, models, grader`.
- [ ] **Step 1** Tests: define a KNOWN-BAD rubric (presence-only criteria whose keywords are easily stuffed) graded by `FakeGrader()` and a KNOWN-GOOD setup graded by `FakeGrader(window=40)` (robust to appended stuffing). Assert `audit(bad).hacking.hackable is True` and `audit(good).hacking.hackable is False`, and `audit(good).trust_score > audit(bad).trust_score`. This reproduces the qualitative 2606.04923 pattern: the tool separates a gameable reward from a robust one.
- [ ] **Step 2** Run → FAIL (until prior tasks landed; here it validates integration).
- [ ] **Step 3** No new src needed if Tasks 1-15 correct; fix any integration gaps surfaced.
- [ ] **Step 4** Run full suite `python3 -m pytest -q` → PASS.
- [ ] **Step 5** Commit: `test: falsification test — tool separates hackable vs robust rewards`.

### Task 17: README + examples + docs polish
**Files:** Create `README.md`, `examples/quickstart.py`, `examples/rubric.yaml`, `examples/grader.fake.yaml`.
**Interfaces:** none (docs).
- [ ] **Step 1** Write `README.md`: the plain-language "what it does" (rubric reward pre-flight check), the paper lineage (RaR 2507.17746, reward-hacking 2605.12474 & 2606.04923, contrast with RewardBench 2506.01937 / RM-Bench 2410.16184), install, quickstart (library + `rrl demo`), the diagnostics list, and an "LLM-as-a-judge users" note. Add `examples/quickstart.py` running `audit` on `load_demo()`. 
- [ ] **Step 2** Run `python3 examples/quickstart.py` → prints a verdict, exits 0.
- [ ] **Step 3** Run `python3 -m pytest -q` once more → PASS.
- [ ] **Step 4** Commit: `docs: README, quickstart, examples`.

---

## Self-Review

**Spec coverage:** §1 problem → README (T17). §2 goals (library+CLI, inference-only, BYO, demo, label-free headline, stats, provider-agnostic) → T3/T4/T13/T15 + constraints. §3 data model → T2/T3. §4 diagnostics A1→T7, A2→T8, A3→T9, B→T10, C→T11; stats layer→T5. §5 report card (verdict, sub-scores, formats) → T12. §6 API surface → T13 (lib) + T15 (CLI). §7 validation → T16. §8 stack → T1 + constraints. §9 milestones map: M0=T1-4+14, M1=T12-13, M2=T6-8, M3=T5+9+10, M4=T11, M5=T16-17. §10 risks (probe quality, cost, monotonic assumption, composite weighting, HealthBench licensing→synthetic data T14) addressed.

**Placeholder scan:** interfaces are concrete; thresholds locked; no TBDs.

**Type consistency:** all `*Result` names, `run_*` signatures, and `ReportCard` fields match the locked interface block and their consuming tasks (T12/T13).

**Open-question resolutions baked in:** name `rubric-reward-lens`/`rrl`; tight rubric-RL core + broad-audience README note; composite trust score + first-class sub-scores.
