## Summary
<!-- What this PR does and why, in 1–3 sentences. Link the task or issue. -->

## Changes
<!-- One bullet per logical commit or change. -->
-

## How it was tested
<!-- Commands run and what they showed. New tests added? -->
- [ ] `bin/wf pytest -q` passes
- [ ] `bin/wf pytest -q -m slow` passes (required if data, models, training or eval changed)

## Results
<!-- Metrics, timings or memory numbers, if any. Validation split only. "n/a" if none. -->

## Checklist
- [ ] Rebased on the latest `origin/main`; history is focused, logical commits (no WIP or fixup commits)
- [ ] Commit messages follow CONTRIBUTING §2
- [ ] Comments explain *why*; public functions have docstrings with shapes and units
- [ ] Tests added or updated in the mirrored `tests/` path
- [ ] Docs updated (README, `docs/`, docstrings) where behavior changed
- [ ] No secrets, data, weights, checkpoints or `runs/` files committed
