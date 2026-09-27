# Manuscript versus actual figure-generating code

Audit target: paper repository commit
`3ed8a5be225d6c415b8ef747aaf27446d2f16cab` on 2026-09-27.
This repository reproduces existing numerical results and figures; it does not
silently change experiments to match newer prose. The paper repository was not
modified while preparing this code release.

| Item | Current manuscript prose | Code / figure actually reproduced |
|---|---|---|
| Simulation default n_bad / bad fraction | 300 / 0.1 | 100 / 0.30 |
| Simulation repetitions | 20 | 10 |
| Simulation center/band | average (caption/text) | median / 25th–75th percentiles |
| Simulation k-means restarts | 10 | 20 |
| Simulation Noalignment clusters | described as same clustering | separate spectral/k-means call |
| Simulation error normalization | Frobenius distance | Frobenius / sqrt(10) |
| MNIST centralized images | 70,000 | fixed balanced subset of 63,100 |
| MNIST image noise | clipped to [0,1] | no clipping |
| MNIST sigma spacing | 0.2 | 0.4 |
| MNIST error band | one standard error | sample standard deviation |
| MNIST y-axis | logarithmic | linear |
| MNIST optimizer | symmetric FastICA | custom simultaneous SVD Varimax, no FastICA package |

The current manuscript's pixel-side rotation form agrees with the basis side
used by these existing MNIST figures. The later sample-side U*S*R experiment
discussed during development is a separate experiment and does not reproduce
these figures; it is intentionally not substituted here.

Varimax objective equivalence under orthogonality/whitening does not establish
that different optimizer implementations have identical finite iterations,
initializations, stopping rules, or outputs. Methods sections should identify
the actual implementations and settings rather than infer them from a related
objective alone.

Before publication, reconcile the prose/captions with the executed protocol,
or explicitly rerun and replace the figures under the desired revised protocol.
The reproducibility commands in this repository correspond to the **right-hand
column** above.
