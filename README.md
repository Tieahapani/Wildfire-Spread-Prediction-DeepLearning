<h1 align="center">wf-spread</h1>

<p align="center">
  <b>Next-day wildfire spread prediction with the Galileo remote-sensing foundation model</b>
</p>

<p align="center">
  <a href="docs/development.md">Setup</a> ·
  <a href="docs/data.md">Data</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## About

wf-spread fine-tunes [Galileo](https://github.com/nasaharvest/galileo), a pretrained Earth-observation encoder, to predict where an active fire will burn tomorrow. Each prediction is a per-pixel fire map for a 64 × 64 km tile.

- **Task:** segmentation on the Next Day Wildfire Spread (NDWS) dataset.
- **Models:** Galileo, fine-tuned three ways (frozen, LoRA, full), against logistic regression, random forest, MLP and U-Net baselines.
- **Evaluation:**
  - PR-AUC on all pixels, and separately on newly burned pixels only.
  - Bootstrap confidence intervals and paired significance tests.
- **Out-of-time test:** the 2021 California fires.

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

`bin/wf` runs any command inside the container with the project environment, for example `bin/wf python`, `bin/wf pytest` or `bin/wf ruff`. Package versions are pinned in [`env/`](env/). The shared environment and data paths are set up as described in [`docs/development.md`](docs/development.md).

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
