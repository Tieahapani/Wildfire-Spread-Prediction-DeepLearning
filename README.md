# wf-spread — Team 7: fine-tuning Galileo for next-day wildfire spread

Plan: `../project-plan.md` · Schedule: `../project-pathway.md`

## Run anything through `bin/wf`
`bin/wf` runs a command inside the NGC PyTorch container (`/srv/sfsu_srv/containers/pytorch-26.09.sif`) with the project venv active. It works the same on spark-61dd and gx10-6d0c.

    cd /srv/sfsu_srv/wf-spread
    bin/wf python -c "import torch; print(torch.cuda.is_available())"
    bin/wf pytest -q            # environment + data smoke tests
    bin/wf bash                 # interactive shell inside the container

## Installing packages
Add the package to `env/requirements.txt`, then run `bin/wf pip install -r env/requirements.txt`, then `bin/wf pip check`.
`env/constraints.txt` pins every package the container ships (torch, numpy, ...), so pip can never replace the NVIDIA build.
Never run `pip install` outside `bin/wf`.

## Layout
    bin/          wf wrapper
    env/          requirements, constraints (container pins), installed versions
    configs/      one YAML per experiment
    src/wfspread/ data (loader, transforms), models (baselines, Galileo wrapper), train, eval (metrics)
    tests/        pytest; run before every sweep
    scripts/      one-off scripts (conversion, downloads)
    third_party/galileo/  official Galileo code at commit 0f0b5b9 (do not edit)
    data/         raw (read-only), processed, models (not in git)
    runs/         logs, checkpoints, W&B, Optuna journals (not in git)
