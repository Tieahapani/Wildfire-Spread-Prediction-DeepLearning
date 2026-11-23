<h1 align="center">wf-spread</h1>

<p align="center">
  <b>Next-day wildfire spread prediction: how much does spatial context help?</b>
</p>

<p align="center">
  <a href="docs/data.md">Data</a> ·
  <a href="docs/models.md">Models</a> ·
  <a href="docs/training.md">Training</a> ·
  <a href="docs/efficiency.md">Efficiency</a> ·
  <a href="docs/evaluation.md">Evaluation</a> ·
  <a href="docs/serving.md">Serving</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## About

wf-spread predicts where an active fire will burn tomorrow, and compares models that see more and more of the area around each pixel. Each prediction is a per-pixel fire map for a 64 × 64 km tile.

- **Task:** segmentation on the Next Day Wildfire Spread (NDWS) dataset.
- **Models:** persistence (tomorrow's fire is today's fire), logistic regression and an MLP on each pixel's 3 × 3 neighborhood, and a U-Net on the whole tile.
- **Evaluation:**
  - PR-AUC on all pixels, and separately on newly burned pixels only.
  - Bootstrap confidence intervals and paired significance tests.

## Getting started

**Requirements:**
- An NVIDIA GPU with CUDA 13.
- [Apptainer](https://apptainer.org) 1.5+.
- The NGC PyTorch 26.09 container.

```bash
git clone https://github.com/Tieahapani/Wildfire-Spread-Prediction-DeepLearning.git wf-spread
cd wf-spread
scripts/setup/dev-setup.sh      # git hooks and rebase defaults
bin/wf pytest -q                # checks GPU, environment and data
```

`bin/wf` runs any command inside the container with the project environment, for example `bin/wf python`, `bin/wf pytest` or `bin/wf ruff`. Package versions are pinned in [`env/`](env/).

## Data

| | Source | License |
|---|---|---|
| NDWS | [Kaggle: fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) | CC BY 4.0 |
| Galileo weights | [Hugging Face: nasaharvest/galileo](https://huggingface.co/nasaharvest/galileo) | MIT |

Layout, checksums and loading rules are in [`docs/data.md`](docs/data.md).

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Coding agents also follow [`AGENTS.md`](AGENTS.md).

## References

- F. Huot et al., "Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data," *IEEE TGRS*, 2022. [doi:10.1109/TGRS.2022.3192974](https://doi.org/10.1109/TGRS.2022.3192974)
- G. Tseng et al., "Galileo: Learning Global & Local Features of Many Remote Sensing Modalities," 2025. [arXiv:2502.09356](https://arxiv.org/abs/2502.09356)

## License

[MIT](LICENSE). Vendored Galileo code keeps its own [MIT license](third_party/galileo/LICENSE).
