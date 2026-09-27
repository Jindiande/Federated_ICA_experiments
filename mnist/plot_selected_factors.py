"""PDF close-up of the user-selected factors from the original sigma=2 panel."""
import json
import argparse
from pathlib import Path
import numpy as np
from plot import ROOT, FIG, plt

FACTORS = [6, 10, 3]
METHODS = [('SF', 'SF-ICA'), ('SRF', 'SRF-ICA')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'reference_results')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/figures')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    ref=np.load(args.results_dir/'reference.npz')['atoms']
    vmax=float(abs(ref).max())
    for sigma in [0.,2.]:
        with np.load(args.results_dir/'run_0'/f'sigma{sigma:.3f}.npz') as d:
            vmax=max(vmax,*(float(abs(d[k+'_aligned']).max()) for k in ['naive_mean','naive_median','srf_noalignment','SF','SRF']))
    limits=[-vmax,vmax]
    with np.load(args.results_dir/'run_0'/'sigma2.000.npz') as data:
        dictionaries = {key: data[key+'_aligned'].copy() for key, _ in METHODS}
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 5.8))
    for row, (key, name) in enumerate(METHODS):
        for col, factor in enumerate(FACTORS):
            ax = axes[row, col]
            im = ax.imshow(dictionaries[key][:, factor-1].reshape(28, 28),
                           cmap='RdBu_r', vmin=limits[0], vmax=limits[1],
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f'Factor {factor}', fontsize=16, pad=12)
            if col == 0:
                ax.set_ylabel(name, rotation=0, ha='right', va='center',
                              fontsize=15, labelpad=16)
    fig.subplots_adjust(left=.18, right=.985, top=.92, bottom=.13,
                        wspace=.10, hspace=.12)
    cax = fig.add_axes([.30, .058, .57, .022])
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-.2,-.1,0,.1,.2])
    name = 'factors_sigma2.000_SF_SRF_factors6_10_3'
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    (out/f'{name}_provenance.json').write_text(json.dumps(dict(
        source='run_0/sigma2.000.npz', factors=FACTORS, methods=[m[0] for m in METHODS],
        sigma=2, run=0, matching='Existing reference-aligned dictionaries, unchanged',
        color_limits=limits, cmap='RdBu_r', interpolation='nearest',
        per_factor_rescaling=False, selection='Factors requested by the user',
        output_format='PDF'), indent=2))
    print(out/f'{name}.pdf')

if __name__ == '__main__':
    main()
