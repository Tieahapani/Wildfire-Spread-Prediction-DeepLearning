# Instructions for coding agents

These instructions apply to any LLM or coding agent working in this repository. The full standard is in `CONTRIBUTING.md`. **Read it before your first change and follow it exactly.** This file is the checklist you must not skip.

## Before changing anything
1. Run `git status`. Work on a feature branch in a personal clone, never on `main`, and never in the shared copy at `/srv/sfsu_srv/wf-spread`.
2. Run `git fetch origin && git rebase origin/main` to start from the latest `main`.
3. Read the code you're about to change, plus its tests and docs.

## While changing code
- Run all Python through `bin/wf` (container + project venv): `bin/wf python`, `bin/wf pytest`, `bin/wf ruff`.
- **Style:**
  - Match the surrounding code.
  - Follow the naming, indexing and shape conventions in CONTRIBUTING §4.
  - Type hints on public functions.
  - Google-style docstrings with shapes, dtypes and units.
- **Comments say *why*, never restate the code.** Use block comments for logical steps. Cite sources for numbers. Leave no commented-out code.
- Tests go in the mirrored path under `tests/`. A bug fix starts with a failing regression test.
- Update docs (README, `docs/`, docstrings) in the same commit as the behavior they describe.

## Before every commit
1. `bin/wf ruff format .` then `bin/wf ruff check .`, both clean.
2. `bin/wf pytest -q` passes. Also run `bin/wf pytest -q -m slow` if you touched data, models, training or evaluation.
3. **Commit size:** one focused, logical commit per change. Not fragmented, not clustered.
4. **Commit message:**
   - Conventional Commits: `type(scope): imperative summary` of 72 characters or fewer.
   - Then a blank line, one or two sentences of why, and `-` bullet points covering what changed and how it was verified.
   - No AI attribution: no `Co-Authored-By` line for an AI tool and no "Generated with" line, in commits or PR descriptions. The hook and CI reject them.

## Before every push
1. `git fetch origin && git rebase origin/main`, then rerun the tests.
2. Push your branch and open a PR using the template. Never push to `main`, and never merge a PR yourself. A team member merges.

## Never
- Never commit secrets, data, weights, checkpoints, `runs/`, or IP addresses. The repo is public.
- Never copy planning docs or personal logs from `/srv/sfsu_srv` into this repo.
- Never modify `data/raw/` or `third_party/`.
- Never `pip install` outside `bin/wf`. Never loosen `env/constraints.txt`.
- Never evaluate on the `test` split during development. Use `val`.
- Never use `--no-verify` to skip the hooks, and never force-push a branch someone else is using.
