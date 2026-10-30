# Contributing to wf-spread

These are the rules for everyone who changes this repository: team members and coding agents alike. They are enforced where possible:
- ruff formats and lints the code;
- git hooks check commit messages and run the tests;
- CI runs on every pull request.

What tooling can't check is reviewed in the PR.

Contents:
1. Workflow
2. Commits
3. Pull requests
4. Code style
5. Comments and docstrings
6. Testing
7. Documentation
8. Experiments and data
9. Dependencies
10. Public-repo safety

---

## 1. Workflow

**Each person works in their own clone.** The shared copy at `/srv/sfsu_srv/wf-spread` holds the shared data, venv and runs. It always stays on `main` and is updated only with `git pull --ff-only`. Nobody edits code there. See `docs/development.md` for setup.

```
git switch main && git pull --rebase           # 1. start from the latest main
git switch -c feat/eval-pr-auc                 # 2. one branch per task
#    ... edit, test, commit (sections 2 and 6) ...
git fetch origin && git rebase origin/main     # 3. rebase onto main again before pushing
bin/wf pytest -q                               # 4. tests pass after the rebase
git push -u origin feat/eval-pr-auc            # 5. push the branch, open a PR
```

- **Branch names:** `<type>/<short-kebab-description>`, using the commit types below, for example `feat/galileo-input-mapping`, `fix/wind-flip-transform`, `exp/unet-sweep`.
- **History stays linear.** Use `git pull --rebase`, never merge `main` into your branch, and resolve conflicts while rebasing.
- **Nobody pushes to `main` directly.** Everything goes through a PR. The pre-push hook blocks direct pushes.
- **Keep branches short-lived.** Open the PR within a few days, and keep a long task's branch rebased daily.

## 2. Commits

### Size: one logical change per commit
- A commit does **one thing completely**: the code, its tests and its doc update together. It builds and passes the tests on its own.
- **Not fragmented.** "wip", "fix typo", "oops" and "try again" never reach a PR branch. Fold them in with `git commit --fixup <sha>`, then `git rebase -i --autosquash origin/main`.
- **Not clustered.** Unrelated changes, such as a new metric plus a loader refactor plus a config tweak, are separate commits.
- **Test of a good split:** could someone revert this one commit without breaking anything else, and does its subject describe all of it?

### Message format (Conventional Commits)

```
<type>(<scope>): <imperative summary, lowercase, no period, ≤ 72 chars>

<Why this change is needed: one or two sentences of context.>

- <What changed, as specific points>
- <Design decisions or trade-offs a reviewer should know>
- <How it was verified: tests added or run, numbers measured>

Refs: #<issue or PR>    (optional)
```

| Type | Use for |
|---|---|
| `feat` | new functionality (model, metric, loader, CLI option) |
| `fix` | bug fix (with a regression test) |
| `perf` | speed or memory improvement, with numbers in the body |
| `refactor` | restructuring with no behavior change |
| `test` | tests only |
| `docs` | documentation only |
| `exp` | experiment configs or sweep definitions |
| `build` | dependencies, container, environment |
| `ci` | CI workflows, git hooks |
| `chore` | maintenance that fits nowhere else |

Scopes match the package areas: `data`, `models`, `galileo`, `train`, `eval`, `serve`, `utils`, `configs`, `env`, `repo`.

Example:

```
feat(eval): add masked PR-AUC and spread-only metrics

Every model's results go through one metrics module, so the -1
(uncertain) pixels must be excluded identically everywhere.

- Add pr_auc() that drops label -1 pixels before scoring
- Add spread_only mask: pixels not burning on day t
- Add paired bootstrap CI over tiles (1,000 resamples, fixed seed)
- Unit tests on hand-built 4x4 cases with known answers
```

The `commit-msg` hook rejects subjects that don't match the format.

## 3. Pull requests

- **Size:** one PR is one task, ideally under about 400 changed lines. Split anything bigger.
- **Description:** fill in the template in `.github/pull_request_template.md`: summary, changes, how it was tested, results, checklist.
- **Before you ask for review:**
  - rebased on `origin/main`
  - `bin/wf pytest -q` passes
  - `bin/wf ruff format --check .` and `bin/wf ruff check .` are clean
  - CI is green
- **Review:** any of the three of us can review any PR. Ask for a review when the change affects someone else's area or the shared pipeline (data, metrics, training core). The reviewer checks logic, tests and the rules here, not just style.
- **Who merges:** any team member can merge any PR, **including their own**, once CI is green and the template checklist is complete. Approval is encouraged but not required. Coding agents never merge; a person does.
- **Merging:** use **"Rebase and merge"**, so each clean commit lands on `main` as-is. Use "Squash and merge" only if the whole PR is one logical change.
- **After merging:** the person who merged deletes the branch, then updates the shared copy with `cd /srv/sfsu_srv/wf-spread && git pull --ff-only`.

## 4. Code style

- **Formatting:** `ruff format`, a Black-compatible style with 100-character lines. Never hand-format and never argue style; the tool decides.
- **Linting:** `ruff check`, with the rules set in `pyproject.toml`. Fix warnings rather than silencing them. A `# noqa: <code>` needs a reason on the same line.
- **Type hints** on every public function signature.
- **No hard-coded paths, magic numbers or hidden globals.** Paths come from configs or the `WF_DATA` / `WF_RUNS` environment variables. Constants live in one place (see Indexing).

### Naming

| Kind | Convention | Example |
|---|---|---|
| modules, packages | `snake_case`, short nouns | `metrics.py`, `wfspread.data` |
| functions, variables | `snake_case`, verbs for functions | `compute_pr_auc`, `fire_mask` |
| classes | `PascalCase` | `NdwsDataset`, `GalileoSegmenter` |
| constants | `UPPER_SNAKE_CASE` | `NDWS_CHANNELS`, `IGNORE_INDEX` |
| private helpers | leading underscore | `_count_records` |
| booleans | `is_`, `has_`, `use_` | `use_lora`, `is_frozen` |
| quantities with units | unit suffix | `wind_dir_deg`, `temp_max_k`, `elapsed_s` |
| configs | `<area>/<name>.yaml`, kebab-case | `configs/model/galileo-tiny.yaml` |
| test files | `test_<module>.py`, mirroring `src/` | `tests/unit/eval/test_metrics.py` |

One name per concept across the whole codebase: `train` / `val` / `test` for splits (NDWS calls validation `eval`; rename it once, in the loader), `prev_fire` / `next_fire` for masks, `tile` for one 64 × 64 sample.

### Indexing and shapes

- **Tensor layout is `(B, C, H, W)`** for images, and `(B, H, W)` for masks. Any other layout is converted at the model boundary and commented.
- **Channel indices come from one named source.** `NDWS_CHANNELS` defines the order, and code uses `CH["wind_dir"]`, never a bare `x[:, 1]`.
- **0-based, half-open ranges** (`[start, stop)`), as in Python.
- **Label values are named constants:** `FIRE = 1`, `NO_FIRE = 0`, `IGNORE_INDEX = -1`.
- **Seeds:** every random source (torch, numpy, Python, data loader workers) is seeded from the run config's `seed`.
- **Run IDs:** `YYYYMMDD-HHMM_<model>_<short-tag>`, for example `20261105-1430_unet_focal-lr3e-4`. The output directory is `runs/<run-id>/`.

## 5. Comments and docstrings

**Comments explain *why*, not *what*.** The code already says what it does. A comment adds intent, a reason, a source, units, or a non-obvious constraint.

```python
# Bad: restates the code
x = x.flip(-1)  # flip x horizontally
theta = (360 - theta) % 360  # subtract theta from 360

# Good: explains the reasoning and the trap
# A horizontal flip mirrors the scene, so wind blowing east now blows west.
# Without this the wind channel silently contradicts the fire pattern.
x = x.flip(-1)
theta = (360 - theta) % 360
```

- **Block comments** above a logical section (3–15 lines that do one step) say what the step is *for*: "Normalize reused bands with Galileo's pretraining statistics so the frozen patch embeddings see familiar ranges."
- **Cite sources** for numbers and choices: `# Huot et al. 2022, Table II`, or `# Galileo normalization.json, commit 0f0b5b9`.
- **No commented-out code, no stale comments.** Delete it; git remembers. When you change code, update its comments in the same commit.
- **`TODO(name): ...`** only with an owner. Open an issue for anything that outlives the PR.

**Docstrings** follow the Google style. Every public module, class and function needs one: a one-line summary, then `Args`, `Returns` and `Raises` where they apply. **State shapes, dtypes and units.**

```python
def pr_auc(probs: Tensor, labels: Tensor) -> float:
    """Area under the precision-recall curve over valid pixels.

    Pixels labelled IGNORE_INDEX (-1, cloud or uncertain) are removed
    before scoring, as in Huot et al. 2022.

    Args:
        probs: Predicted fire probability, shape (B, H, W), float in [0, 1].
        labels: Next-day fire mask, shape (B, H, W), values in {-1, 0, 1}.

    Returns:
        PR-AUC in [0, 1].

    Raises:
        ValueError: If no valid fire pixels remain.
    """
```

## 6. Testing

**Every change is tested before it's committed, and the full suite passes again before every push and after every rebase.**

| Folder | What | Speed |
|---|---|---|
| `tests/unit/` | Pure logic on small hand-built inputs with known answers: metrics, transforms, shapes, config parsing | seconds, CPU |
| `tests/integration/` | Real pieces together: environment, GPU, data files, one batch through each model | under 1 minute |
| `tests/regression/` | Published or previously measured numbers must not drift, e.g. persistence PR-AUC 0.115 ± 0.01 and LR 0.198 ± 0.01, and frozen metric outputs on a fixed batch | minutes; marked `slow` |

- **Bug fix = regression test first.** Write a test that fails on the bug, then fix it.
- **New code ships with tests** in the same commit, in the mirrored path (`src/wfspread/eval/metrics.py` → `tests/unit/eval/test_metrics.py`).
- **Markers:** `gpu` (needs CUDA), `data` (needs `WF_DATA`), `slow` (over 30 s). The default run `bin/wf pytest -q` runs everything except `slow`. Run `bin/wf pytest -q -m slow` before merging anything that touches data, models, training or evaluation.
- **Deterministic tests:** fixed seeds, no network access, no dependence on test order.
- **Numbers in the report come from tested code paths only.**

The `pre-commit` hook runs ruff on staged files. The `pre-push` hook runs the default test suite and blocks the push if anything fails.

## 7. Documentation

- **`README.md`:** what the project is, quick start, layout. Update it when commands or layout change.
- **`docs/`:** longer documentation: `development.md` (setup, workflow), `data.md` (datasets, provenance, licenses), and further pages as features land, e.g. `models.md`, `experiments.md`.
- **Docstrings** document the API. `docs/` documents how the pieces fit together and why.
- **Configs are self-describing:** a short header comment per YAML file says what the experiment tests.
- **Docs change in the same commit as the behavior they describe.**
- Planning documents, personal logs and course material **stay outside this repo** and are never committed.

## 8. Experiments and data

- **`data/raw/` is read-only.** Derived files go in `data/processed/`, written by a script in `scripts/data/` so they can be regenerated.
- **Every run comes from a config file**, never from ad-hoc command-line edits. The config, git commit SHA, seed and environment are logged with the run.
- **The test split is touched only for final evaluation, by the evaluation owner, once per final model.** All tuning and model selection uses `val`.
- **Outputs** (checkpoints, logs, Optuna journals, W&B files) go in `runs/` and are never committed.
- **Optuna** uses `JournalFileBackend` storage, never SQLite on the shared NFS folder.

## 9. Dependencies

- Add the package to `env/requirements.txt`, then `bin/wf pip install -r env/requirements.txt`, then `bin/wf pip check`, then update `env/project-installed.txt` (`bin/wf pip freeze`). Commit all of it as one `build(env): ...` commit.
- `env/constraints.txt` pins everything the NVIDIA container ships. **Never loosen it to make an install work.** Pick a compatible version of the new package instead.
- Never run `pip install` outside `bin/wf`. Never edit `third_party/`; wrap it in `src/wfspread/models/` instead.

## 10. Public-repo safety

This repository is **public**.
- **Never commit:** keys, tokens, passwords, `.env` files, the W&B API key, IP addresses, data, model weights, or checkpoints.
- Check `git status` and `git diff --cached` before every commit.
- If a secret is committed, tell the team immediately and rotate it. Removing it from history is not enough.
