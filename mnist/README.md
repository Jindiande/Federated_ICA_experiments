# MNIST paper-figure pipeline

## Run and plot

```sh
# From repository root; download is automatic if data are missing:
python mnist/run.py
python mnist/plot.py --results-dir mnist/outputs
python mnist/plot_selected_factors.py --results-dir mnist/outputs
```

For exact figure inputs without retraining, omit `--results-dir`; plotting then
uses `reference_results/`. Full retraining never reads those saved aggregate
dictionaries or metric values.

## Files

| File | Purpose |
|---|---|
| `run.py` | Full 5-partition, 6-noise-level, 5-method experiment |
| `data.py` | Download/verify original MNIST IDX gzip files; concatenate train then test |
| `varimax.py` | Original simultaneous orthogonal Varimax SVD updates |
| `aggregation.py` | Spectral embedding, k-means, sign alignment, mean/GM aggregation |
| `dictionary_metrics.py` | Unit-column Hungarian matching and signed error |
| `plot.py` | Error curve and both full factor panels |
| `plot_selected_factors.py` | SF/SRF enlarged factors, ordered 6,10,3 |
| `config/partitions.npz` | Exact original clean pool, 5 client partitions, bad-client IDs |
| `config/experiment.json` | All initialization seeds and input hashes for 151 fits |
| `config/data_manifest.json` | Public download URLs and SHA-256 checksums |
| `config/reference_display.npz` | Centralized-only historical factor display orientation |

## Exact settings behind the current paper PDFs

- Source pool: 70,000 MNIST train+test images. Experiments use the fixed balanced
  **63,100-image subset**, including the centralized reference; not all 70,000.
- Ten disjoint clients per run, 6,310 images each, exactly 631 per digit.
- Five original partitions; four designated noisy clients and six clean clients.
- Sigma = **0,0.4,0.8,1.2,1.6,2.0**.
- Noisy client input: `uint8_images/255 + sigma * Z`, Z iid standard Gaussian.
  **No clipping** and no noise on fitted dictionary entries. Z is fixed across
  sigma for each client/run. Its seed is generated with NumPy SeedSequence from
  `[20260912,101,20,run,client]`.
- Each input is centered by its local per-pixel sample mean. PCA retains r=10.
- For conceptual pixel-by-image SVD `X=U*S*V.T`, Varimax operates on **U**,
  giving **A=U*R**. No singular-value weights are included. These are the
  orthogonal dictionaries behind the archived figures, not the later U*S*R run.
- Varimax uses no Kaiser row normalization, global RMS scaling only, 6000
  maximum iterations, rotation-step tolerance 1e-8. Central: identity plus
  100 random starts. Local: 30 original seeds. Choose the highest-objective
  converged start. This is a simultaneous SVD/polar Varimax iteration, not a
  FastICA package or deflation-Varimax implementation.
- Local signs/order are raw optimizer outputs. There are no artificial sign
  flips and no reference pairing before server aggregation.
- Server: absolute Gram affinity, top-10 eigenvectors weighted by eigenvalues,
  20-start k-means; shared clusters for SF, SRF and Noalignment. Within-cluster
  leading singular vectors determine SF/SRF sign alignment.
- Naive Mean uses raw column-index means. **Naive Median uses geometric median
  by raw column index**, unlike the simulation's coordinate-wise median.
- All MNIST geometric medians use tolerance 1e-7, cap 30,000 iterations.
- Evaluation: normalize columns, Hungarian match using absolute inner products,
  correct signs, Frobenius difference / sqrt(10). The central reference is an
  empirical benchmark, not population ground truth.

The central display template only establishes the original reference column
order/sign convention after fitting. It does not replace the newly fitted
central estimate, initialize local fits, or guide server clustering.

## Figure conventions

Error curve: mean ± **sample standard deviation** over five partitions, **linear**
y-axis. Factors: fixed run 0, shared signed `RdBu_r` limits across sigma=0 and 2,
no per-image rescaling. Selected factors use those same color limits.

To match the current paper PDFs exactly, names retain the requested historical
display variants: the error legend uses `SRF-ICA-Noalignment`, while the two
factor panels use `SRF-Noalignment-ICA`; both mean the same method. All other
SF/SRF labels end in `-ICA`.

## Fresh-run outputs

`outputs/fits/` contains each newly estimated dictionary and initialization logs;
`outputs/run_*/` contains partitions and per-sigma local/aggregate dictionaries;
`metrics.csv` has 150 rows and `summary.csv` contains mean/SD summaries.
Input hashes are checked for every fit. `--resume` only reuses matching fits
inside the requested output directory. The default run does all 151 fits fresh.
