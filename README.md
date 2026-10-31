# wf-spread: fine-tuning Galileo for next-day wildfire spread

SFSU CSC 671/871 (Fall 2026), Team 7.

We fine-tune Galileo, an open-weight Earth-observation foundation model, to predict next-day wildfire spread as per-pixel segmentation on Google's Next Day Wildfire Spread (NDWS) dataset. We compare it with from-scratch and ImageNet-pretrained networks, and add a case study on 2021 California fires.

## Quick start

```bash
cd /srv/sfsu_srv/dev/$USER/wf-spread     # your personal clone (see docs/development.md)
bin/wf pytest -q                         # environment, GPU and data checks
bin/wf python -c "import torch; print(torch.cuda.get_device_name(0))"
```

`bin/wf` runs a command in the NGC PyTorch 26.09 container with the shared project venv, on either Spark.

## Project layout

```
wf-spread/
├── .github/            PR template, CI workflow
├── .githooks/          pre-commit, commit-msg, pre-push
├── bin/wf              run anything inside the container + venv
├── configs/            YAML per experiment: data/, model/, experiment/
├── docs/               development.md, data.md (more pages as features land)
├── env/                requirements.txt, constraints.txt (container pins), installed versions
├── notebooks/          exploration only; nothing the report depends on
├── scripts/            entry points: setup/, data/, train/, eval/
├── src/wfspread/       the package
│   ├── data/           NDWS loading, conversion, transforms
│   ├── models/         baselines and the Galileo wrapper
│   ├── train/          shared training module and loops
│   ├── eval/           metrics, bootstrap, evaluation runs
│   ├── serve/          inference and serving
│   └── utils/          seeding, logging, config helpers
├── tests/              unit/, integration/, regression/ (mirrors src/)
├── third_party/galileo official Galileo code at commit 0f0b5b9 (do not edit)
├── data/  runs/        shared datasets and outputs (not in git)
└── pyproject.toml      package metadata, ruff and pytest config
```

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md). It covers branches, commits, PRs, code style, comments, tests and documentation. Coding agents follow [`AGENTS.md`](AGENTS.md). Claude Code, Gemini CLI and GitHub Copilot load it through `CLAUDE.md`, `GEMINI.md` and `.github/copilot-instructions.md`; Codex and Cursor read it directly.

## Data and licenses

See [`docs/data.md`](docs/data.md). Code: MIT. NDWS: CC BY 4.0. Galileo: MIT.
