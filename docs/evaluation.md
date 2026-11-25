# Final evaluation

`scripts/eval/final_eval.py` scores the final models listed in `configs/eval/final.yaml` and writes the numbers for the results table.

## What it reports

For every model, each run (one per seed) is scored at its own decision threshold, chosen for best F1 on the **validation** split and then applied unchanged to the scored split. Persistence (tomorrow's fire is today's fire) needs no training and is scored the same way.

| Output | Meaning |
|---|---|
| mean ± std | Spread over the seeds: how much training randomness moves the score |
| 95% CI | Tile bootstrap of the seed-mean PR-AUC (1,000 resamples): how much the score depends on which tiles were sampled |
| Paired comparison | Bootstrap of the difference between two models on the same resampled tiles, with a one-sided p-value for "A is better" |

Tiles are resampled whole, never pixels, because pixels in a tile are strongly correlated. See `wfspread.eval.metrics.bootstrap_ci` and `paired_bootstrap`.

Results go to `$WF_RUNS/final-eval/<split>/`: `per_run.csv`, `summary.md` and `results.json`.

## Reading the results

- **Validation scores of tuned models are slightly optimistic.** The Optuna search picked its settings by validation PR-AUC, and the final runs are scored on the same validation tiles, so part of any lead may be luck in that choice. The test split, which no choice has seen, gives the unbiased number. Report validation and test side by side.
- **Error analysis groups describe where errors happen, not what can be predicted.** Groups such as fire size or growing versus shrinking fires (`scripts/eval/error_analysis_unet.py`) are defined with the day-t+1 fire mask, which a model never has. They show which kinds of tiles the model gets wrong; they are not a way to flag hard tiles in advance.

## Comparison with Huot et al. 2022

Huot et al. 2022 excluded the 1-pixel tile border when scoring their baselines. `--exclude-border` scores the same way, writing to `<split>-no-border/`:

```bash
bin/wf python scripts/eval/final_eval.py --exclude-border
```

On validation (Nov 24) the border changes almost nothing: logistic regression 0.114 → 0.115 PR-AUC, persistence 0.060 → 0.061. So the border does not explain why our baselines score below the published ones (Huot, test split: persistence 0.115, precision 35.7%, recall 27.3%; logistic regression 0.198):

| Model | PR-AUC, ours (val, no border) | Precision / recall, ours | Huot et al. (test) |
|---|---|---|---|
| Persistence | 0.061 | 0.262 / 0.187 | 0.115; 0.357 / 0.273 |
| Logistic regression | 0.115 ± 0.000 | 0.227 / 0.341 (mean of 3 runs) | 0.198; 0.325 / 0.353 |

Persistence has no training, so its gap comes from the data, not the model: the split (our validation vs their test) or the cleaning. Our persistence on test, scored once at the freeze, will show which. Logistic regression also differs in training (all pixels with fire weighted 3, against their 1:1 balanced set with "no fire" weighted by W = 7).

## Running

During development, score validation only:

```bash
bin/wf python scripts/eval/final_eval.py
```

## The test run, once, at the freeze

The test split is scored once, after every final model is listed in the manifest and the team has agreed on it. The script refuses to score test without `--final-test` or when test results already exist.

```bash
bin/wf python scripts/data/clean.py --splits test        # clean test with training statistics
bin/wf python scripts/eval/final_eval.py --split test --final-test
```

Cleaning the test split only applies the saved training statistics; no number from it is looked at before the run above.

## Adding a model

Add an entry under `models` with its model config and one run folder per seed, relative to `$WF_RUNS`. An ablation that hides input channels lists them in `drop_channels`; the model is then built with fewer input channels and wrapped in `wfspread.models.wrappers.ChannelSubset`, matching how it was trained. A model that is not a trained network needs its own `kind`.
