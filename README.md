# Federated ICA: simulation and MNIST figure reproduction

This is the code repository for [Federated Independent Component Analysis via Spectral Alignment and Robust Aggregation](https://arxiv.org/abs/2505.20532).

## Install

Python 3.9–3.12; exact package versions are pinned in `requirements.txt`.
The original Python environment was Python 3.9 with NumPy 1.26.4.

```sh
git clone https://github.com/Jindiande/Federated_ICA_experiments.git
cd Federated_ICA_experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Retraining the simulation additionally requires **MATLAB with Statistics and
Machine Learning Toolbox** (original version: R2026a). MATLAB must be on PATH;
otherwise replace `matlab` with its executable's absolute path.
Replotting saved simulation results does not require MATLAB.

## Reproduce every figure quickly

These commands use the included numerical results. They do not train models or
download MNIST. Run from the repository root:

```sh
python simulation/plot_results.py
python mnist/plot.py
python mnist/plot_selected_factors.py
python tools/verify_results.py
```

Only PDFs are exported. Plot scripts read `reference_results/` by default and
write to `outputs/`; archived results are never overwritten by these commands.

## Which file reproduces which figure?

Run the following commands from the repository root.

| Figure | Command | Generated PDF |
|---|---|---|
| Simulation: three panels | `python simulation/plot_results.py` | `simulation/outputs/figures/five_methods_threepanels.pdf` |
| MNIST error curve | `python mnist/plot.py` | `mnist/outputs/figures/recovery_error_five_methods.pdf` |
| MNIST factors at σ=0 | `python mnist/plot.py` | `mnist/outputs/figures/factors_sigma0.000.pdf` |
| MNIST factors at σ=2 | `python mnist/plot.py` | `mnist/outputs/figures/factors_sigma2.000.pdf` |
| MNIST enlarged factors 6,10,3 | `python mnist/plot_selected_factors.py` | `mnist/outputs/figures/factors_sigma2.000_SF_SRF_factors6_10_3.pdf` |

The selected-factor plot is independent: it computes its common color limits
from the saved σ=0 and σ=2 dictionaries, so running `plot.py` first is optional.
Reference PDFs are preserved in `paper_figures/`; their checksums are recorded
in `provenance.json`. PDF creation timestamps may differ
on regeneration; numerical and rendered-figure agreement matter instead.

## Rerun the complete experiments

Simulation (20 settings × 10 repetitions, all five methods):

```sh
matlab -batch "run('simulation/main_2.m')"
python simulation/plot_results.py --results-dir simulation/outputs
```

MNIST (central fit + 50 clean local fits + 100 noisy local fits):

```sh
python mnist/run.py
python mnist/plot.py --results-dir mnist/outputs
python mnist/plot_selected_factors.py --results-dir mnist/outputs
python tools/verify_results.py --mnist-results mnist/outputs --simulation-results simulation/outputs
```

`mnist/run.py` downloads the four original MNIST gzip files when missing and
verifies SHA-256 checksums. Alternatively use
`python mnist/run.py --data-dir /path/to/mnist-gzip-files`.
Use `--resume` to reuse validated fits from an interrupted run's own output
directory. No sibling repositories, private datasets, absolute user paths, or
pretrained local dictionaries are needed. Exact original partitions and seed
lists are included as experimental inputs in `mnist/config/`.

Full reruns of nonconvex optimization can vary across numerical libraries and
platforms. In particular, raw signs/order influence the deliberately unaligned
baselines. The archived results provide the exact inputs for reproducing the
published visualizations. Verification reports fresh-run discrepancies rather
than silently replacing fresh results with archived values.

## Layout and protocol

- [simulation/README.md](simulation/README.md): MATLAB estimator, sweep settings,
  RNG, error metric, median/IQR plotting.
- [mnist/README.md](mnist/README.md): MNIST input selection, Varimax implementation,
  noise generation, aggregation, mean/SD plotting, reference orientation.
- `validation.json`: checks actually completed when this repository was prepared.
- `tools/verify_results.py`: reusable numerical verification.

The code has been reviewed by Xin Bing, Dian Jin and Yuqian Zhang. Please contact uestcjd@gmail.com if there is any problem.
