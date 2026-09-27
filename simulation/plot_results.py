"""Three-panel simulation figure using the MNIST comparison's visual style.

Run main_2.m in MATLAB first, then python plot_results.py.
The default vertical axis is logarithmic, as in the paper.
Curves show medians; shaded bands show the 25th–75th percentiles.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/neurips26-simulation-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
METHODS = ['naive_mean', 'naive_median', 'srf_noalignment', 'SF', 'SRF']
SOURCE_INDEX = {'SRF': 0, 'srf_noalignment': 1, 'SF': 2,
                'naive_mean': 3, 'naive_median': 4}
NAMES = {'naive_mean': 'Naive Mean', 'naive_median': 'Naive Median',
         'srf_noalignment': 'SRF-ICA-Noalignment', 'SF': 'SF-ICA', 'SRF': 'SRF-ICA'}
COLORS = {'naive_mean': '#777777', 'naive_median': '#b08a39',
          'srf_noalignment': '#8452a1', 'SF': '#ce7130', 'SRF': '#2462a5'}
STYLES = {'naive_mean': ('--', 's'), 'naive_median': (':', 'v'),
          'srf_noalignment': ('-.', '^'), 'SF': ('-', 'o'), 'SRF': ('-', 'D')}
SPECS = {
    'varyK': ('(a) Number of clients', r'Number of clients, $K$',
              [10, 30, 50, 70, 100]),
    'varyNbad': ('(b) Low-sample client size',
                 r'Samples per low-sample client, $n_{\mathrm{bad}}$',
                 [50, 70, 100, 300, 500, 1000]),
    'varyBadFrac': ('(c) Low-sample client fraction',
                    r'Low-sample client fraction, $\rho$',
                    np.arange(9) * .05),
}


def read_results(results_dir, output_dir):
    result = loadmat(results_dir / 'simulation_results.mat',
                     simplify_cells=True)['results']
    assert result['repeats'] == 10
    assert list(result['method_names']) == [
        'SRF-ICA', 'SRF-ICA-Noalignment', 'F-ICA', 'simple mean', 'simple median']
    params = result['base_params']
    expected = dict(r=10, theta=.1, noise_std=0, n_good=5000,
                    n_bad=100, bad_frac=.3, K=30)
    assert all(params[k] == v for k, v in expected.items())
    experiments = result['all_results']
    assert [e['exp_mode'] for e in experiments] == list(SPECS)
    rows = []
    for e in experiments:
        mode, x, raw = e['exp_mode'], e['sweep'], e['errors_repeats']
        np.testing.assert_allclose(x, SPECS[mode][2], atol=1e-14)
        assert raw.shape == (10, 5, len(x))
        assert np.isfinite(raw).all() and (raw >= 0).all()
        assert (raw <= np.sqrt(2) + 1e-10).all()
        np.testing.assert_allclose(raw.mean(axis=0), e['errors'], atol=1e-14)
        np.testing.assert_allclose(raw.std(axis=0, ddof=1), e['errors_std'], atol=1e-14)
        for t, value in enumerate(x):
            for rep in range(10):
                for key in METHODS:
                    rows.append(dict(experiment=mode, value=float(value),
                                     repeat=rep + 1, method=key,
                                     error=float(raw[rep, SOURCE_INDEX[key], t])))
    df = pd.DataFrame(rows)
    assert len(df) == 1000
    df.to_csv(output_dir / 'metrics.csv', index=False)
    summary = df.groupby(['experiment', 'value', 'method'], sort=False).error.agg(
        count='count', mean='mean', std='std', median='median',
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)).reset_index()
    summary.to_csv(output_dir / 'summary.csv', index=False)
    (output_dir / 'verification.json').write_text(json.dumps(dict(
        complete=True, settings=20, repeats_per_setting=10, method_evaluations=len(df),
        parameters=expected, mean_and_sample_sd_verified=True,
        plot_center='median', plot_band='25th–75th percentiles',
        quantile_interpolation='linear, computed on raw errors',
        original_algorithm_function_bodies_unchanged=True,
        source_commit='6e96400', elapsed_seconds=result['elapsed_seconds'],
        matching='Original MATLAB matchpairs; absolute similarities, final sign correction',
        naive_median='Coordinate-wise median, as in the source repository',
        noalignment='Separate spectral clustering call, as in the source repository',
    ), indent=2) + '\n')
    return experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yscale', choices=['linear', 'log'], default='log')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'reference_results')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiments = read_results(args.results_dir, args.output_dir)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2), sharey=True)
    floor = 1e-3
    for ax, e in zip(axes, experiments):
        mode = e['exp_mode']
        x = e['sweep']
        for key in METHODS:
            i = SOURCE_INDEX[key]
            low, median, high = np.quantile(
                e['errors_repeats'][:, i, :], [.25, .5, .75], axis=0,
                method='linear')
            assert np.all((low > floor) & (low <= median) & (median <= high))
            line, marker = STYLES[key]
            ax.plot(x, median, ls=line, marker=marker, color=COLORS[key],
                    label=NAMES[key], lw=2, ms=5)
            ax.fill_between(x, low, high, color=COLORS[key], alpha=.12)
        span = x[-1] - x[0]
        ax.set(xlim=(x[0] - .04 * span, x[-1] + .04 * span),
               xlabel=SPECS[mode][1], yscale=args.yscale)
        ax.set_title(SPECS[mode][0], fontsize=12, pad=12)
        if mode == 'varyK':
            ax.set_xticks(x)
        elif mode == 'varyNbad':
            ax.set_xticks([50, 300, 500, 1000])
        else:
            ax.set_xticks(np.arange(5) * .1)
        ax.grid(alpha=.18)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=14)
    axes[0].set_ylabel('Matched dictionary error (log scale)' if args.yscale == 'log'
                       else 'Matched dictionary error')
    axes[0].set_ylim((floor, 1.12) if args.yscale == 'log' else (-.07, 1.07))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.01),
               ncol=5, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.06, right=.99, bottom=.16, top=.8, wspace=.12)
    out = args.output_dir / 'figures'
    out.mkdir(exist_ok=True)
    name = 'five_methods_threepanels' + ('_linear' if args.yscale == 'linear' else '')
    for ext in ['pdf']:
        fig.savefig(out / f'{name}.{ext}', dpi=200,
                    bbox_inches='tight', facecolor='white')
        if args.yscale == 'log':
            shutil.copyfile(out / f'{name}.{ext}', out / f'{name}_log.{ext}')
    plt.close(fig)
    print(f'Validated 1,000 method evaluations; saved {out / name}.pdf')


if __name__ == '__main__':
    main()
