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

Persistence has no training, so its gap had to come from the data. The test run settles it: on test, persistence scores precision 0.357 and recall 0.273, exactly Huot's 35.7% and 27.3%, so our cleaning matches theirs and the gap was the split. Validation is simply a harder split for these baselines. Logistic regression on test is 0.204 ± 0.001 without the border (Huot: 0.198), and persistence 0.108 (Huot: 0.115; the PR-AUC calculation for yes/no predictions is not stated in their paper).

## Test results (Nov 25, scored once)

The test split (1,689 tiles) was cleaned with the training statistics and scored once with `--final-test`, at thresholds chosen on validation. Outputs are in `$WF_RUNS/final-eval/test/` and `test-no-border/`.

| Model | PR-AUC (95% CI) | F1 | IoU | Spread PR-AUC |
|---|---|---|---|---|
| Persistence | 0.107 (0.097–0.117) | 0.309 | 0.183 | 0.009 |
| Logistic regression | 0.202 ± 0.001 (0.186–0.220) | 0.334 ± 0.002 | 0.201 ± 0.001 | 0.090 ± 0.000 |
| MLP 3 × 3 | 0.232 ± 0.003 (0.217–0.249) | 0.346 ± 0.002 | 0.209 ± 0.001 | 0.104 ± 0.001 |
| **U-Net (final)** | **0.351 ± 0.004** (0.336–0.367) | 0.406 ± 0.007 | 0.255 ± 0.006 | 0.255 ± 0.011 |
| U-Net, tuned | 0.355 ± 0.008 (0.341–0.372) | 0.401 ± 0.009 | 0.251 ± 0.007 | 0.263 ± 0.009 |
| U-Net, tuned, no today's fire mask | 0.209 ± 0.008 (0.200–0.218) | 0.273 ± 0.015 | 0.158 ± 0.010 | 0.174 ± 0.008 |

Paired differences in PR-AUC, first model minus second, with 95% tile-bootstrap intervals:

| Comparison | All pixels | Spread region |
|---|---|---|
| U-Net − MLP | +0.118 (+0.108 to +0.129) | +0.151 (+0.140 to +0.163) |
| MLP − logistic regression | +0.030 (+0.023 to +0.037) | +0.014 (+0.011 to +0.017) |
| Logistic regression − persistence | +0.096 (+0.088 to +0.104) | +0.081 (+0.075 to +0.087) |
| Tuned U-Net − U-Net | +0.005 (+0.001 to +0.008) | +0.008 (+0.004 to +0.013) |
| Tuned U-Net − same without today's fire mask | +0.147 (+0.132 to +0.162) | +0.089 (+0.077 to +0.103) |

Each step up in spatial context helps on test too. The tuned U-Net is slightly but measurably better on test (+0.005), unlike on validation (+0.002, interval including 0). The final model was chosen on validation before the test run and stays the small U-Net; the tuned result is reported as a small gain for 4.7 times the parameters.

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

## Report figures

`scripts/eval/report_figures.py` draws precision-recall curves and example maps (today's fire, logistic regression, MLP and U-Net probabilities, tomorrow's fire) from each model's seed-0 run. Example tiles are drawn at random, with a fixed seed, from tiles with at least 20 fire pixels tomorrow, so they are not picked by eye. Figures go to `$WF_RUNS/final-eval/figures/`.

```bash
bin/wf python scripts/eval/report_figures.py --split test
```

Curves are drawn as steps, matching how PR-AUC is computed; joining persistence's single operating point to the corners with straight lines would overstate it. On test, logistic regression and the MLP lose most of their precision at about recall 0.47. That is the share of tomorrow's fire within one pixel of today's fire (47.3%; 27.3% is fire that is already burning). A 3 × 3 model cannot see fire farther away, so beyond that recall it has little to go on, while the U-Net keeps finding fire that spreads farther (58.1% of tomorrow's fire is within two pixels, 64.6% within three).

`scripts/eval/training_curves.py` draws training loss, validation loss and validation PR-AUC per epoch for every seed of logistic regression, the MLP and the U-Net, with median epoch time and peak GPU memory, from the runs' `logs/metrics.csv`. It saves `training-curves.png` in the same folder.

## Adding a model

Add an entry under `models` with its model config and one run folder per seed, relative to `$WF_RUNS`. An ablation that hides input channels lists them in `drop_channels`; the model is then built with fewer input channels and wrapped in `wfspread.models.wrappers.ChannelSubset`, matching how it was trained. A model that is not a trained network needs its own `kind`.
