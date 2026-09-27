# Simulation

## Entry points

- `main_2.m`: full synthetic-data generation, all local estimation, five server
  methods, and matched recovery errors. All helper functions are in this file.
- `plot_results.py`: validates individual repetition results, writes CSV summaries,
  and generates the three-panel PDF. `--results-dir` selects saved or fresh data.

Run from the repository root as documented in the root README.

## Actual experiment settings

| Setting | Value |
|---|---|
| Dimension/rank | 10 |
| Source law | Bernoulli(0.1) × standard Gaussian |
| Observation noise | 0 |
| Good-client sample size | 5000 |
| Default clients / bad-client sample size / fraction | 30 / 100 / 0.30 |
| Left sweep | K = 10,30,50,70,100 |
| Middle sweep | n_bad = 50,70,100,300,500,1000 |
| Right sweep | bad fraction = 0,0.05,...,0.40 |
| Repetitions | 10 per setting, 200 total |
| RNG | MATLAB `rng(1)`, one sequential stream in the original loop order |

Each repetition redraws the ground-truth dictionary and all client data.
Low-sample clients are the first `round(bad_frac*K)` clients. Do not reorder
sweep/repetition/method loops if reproducing the same RNG stream.

The original local estimator performs `Y = U*S*V'`, calls MATLAB
`rotatefactors(V(:,1:r), 'Method','orthomax','Normalize','off','Maxit',1000)`,
and returns column-normalized `U(:,1:r)*T`. It does **not** multiply by S.
This is retained exactly from the source, not replaced with another ICA solver.

SF-ICA and SRF-ICA share spectral clustering and sign alignment. The
Noalignment control retains the source's **separate** spectral clustering
call. K-means uses 20 replicates. Naive Median is **coordinate-wise median**;
the robust within-cluster aggregator is geometric median, tolerance 1e-8,
maximum 500 iterations. These details intentionally differ from MNIST.

The error is signed-permutation-matched Frobenius distance divided by sqrt(10),
using MATLAB `matchpairs`. Statistics and Machine Learning Toolbox is required;
the source includes fallbacks, but those are not the validated paper pipeline.

## Plot and provenance

Curves are **medians**, bands are **25th–75th percentiles**, and the y-axis is
logarithmic. Quantiles are computed on the raw errors. The figure is not
mean ± SD/SE. Python display names map the source's `F-ICA` to `SF-ICA`.

`reference_results/simulation_results.mat` contains all errors, parameters,
and repetition-level results. `main_2.m` writes fresh data under `outputs/`.
Its `checkpoint.mat` is diagnostic only; the MATLAB script does not resume
from that checkpoint. `plot_results.py --yscale linear` is optional and writes
a separate `_linear.pdf`, not the paper's default figure.

Original source: https://github.com/Jindiande/neurips_26_seceret,
commit `6e96400cc30744a0032599ce1eb9d54d482cc08a`. Algorithm function bodies are
preserved. The runner additionally saves individual repetitions and has a
portable output path. Original source checksums are included in
`reference_results/provenance.json`.
