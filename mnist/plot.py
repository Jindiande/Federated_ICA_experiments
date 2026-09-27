"""Image-noise version of the five-method recovery and factor figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mnist-image-noise-mpl')
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'outputs/figures'
METHODS=['naive_mean','naive_median','srf_noalignment','SF','SRF']
NAMES={'naive_mean':'Naive Mean','naive_median':'Naive Median','srf_noalignment':'SRF-ICA-Noalignment','SF':'SF-ICA','SRF':'SRF-ICA'}
COLORS={'naive_mean':'#777777','naive_median':'#b08a39','srf_noalignment':'#8452a1','SF':'#ce7130','SRF':'#2462a5'}
STYLES={'naive_mean':('--','s'),'naive_median':(':','v'),'srf_noalignment':('-.','^'),'SF':('-','o'),'SRF':('-','D')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    global FIG
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'reference_results')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/figures')
    args=parser.parse_args();data_root=args.results_dir;FIG=args.output_dir;FIG.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data_root/'metrics.csv');sigmas=sorted(df.sigma.unique());assert len(df)==25*len(sigmas)
    max_sigma=float(max(sigmas))
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    for key in METHODS:
        g=df[df.method.eq(key)].groupby('sigma').recovery_error.agg(['mean','std'])
        x=g.index.to_numpy();m=g['mean'].to_numpy();s=g['std'].to_numpy();style,marker=STYLES[key]
        ax.plot(x,m,ls=style,marker=marker,color=COLORS[key],label=NAMES[key],lw=2,ms=5)
        ax.fill_between(x,m-s,m+s,color=COLORS[key],alpha=.12)
    ax.set(xlim=(-.02*max_sigma,1.02*max_sigma),xlabel=r'Training-image noise per pixel, $\sigma$',ylabel='Matched dictionary error')
    ax.set_xticks(sigmas);ax.tick_params(axis='x',labelsize=14);ax.grid(alpha=.18)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.015),ncol=3,fontsize=9.5,frameon=False)
    fig.tight_layout();save(fig,'recovery_error_five_methods')
    gallery={}
    ref=np.load(data_root/'reference.npz')['atoms']
    for sigma in [0,max_sigma]:
        with np.load(data_root/'run_0'/f'sigma{sigma:.3f}.npz') as d:gallery[sigma]={'centralized':ref,**{key:d[key+'_aligned'] for key in METHODS}}
    vmax=max(float(abs(a).max()) for g in gallery.values() for a in g.values())
    for sigma,g in gallery.items():
        fig,axes=plt.subplots(6,10,figsize=(15.8,8.0))
        for row,key in enumerate(['centralized']+METHODS):
            for j in range(10):
                ax=axes[row,j];im=ax.imshow(g[key][:,j].reshape(28,28),cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                for s in ax.spines.values():s.set_visible(False)
                if row==0:ax.set_title(f'Factor {j+1}',fontsize=11,pad=8)
                if j==0:
                    label='Centralized\n(reference)' if key=='centralized' else ('SRF-Noalignment-ICA' if key=='srf_noalignment' else NAMES[key])
                    err=0 if key=='centralized' else float(df[df.run.eq(0)&df.sigma.eq(sigma)&df.method.eq(key)].recovery_error.iloc[0])
                    ax.set_ylabel(label+f'\nerror = {err:.3f}',rotation=0,ha='right',va='center',labelpad=13,fontsize=10,color='#222222' if key=='centralized' else COLORS[key])
        fig.subplots_adjust(left=.14,right=.985,top=.96,bottom=.065,wspace=.055,hspace=.2)
        cax=fig.add_axes([.38,.025,.38,.014]);fig.colorbar(im,cax=cax,orientation='horizontal')
        save(fig,f'factors_sigma{sigma:.3f}')
    (FIG/'figure_protocol.json').write_text(json.dumps(dict(data_source='Only this rerun directory',factor_run=0,color_limits=[-vmax,vmax],methods=METHODS,reference_curve=False),indent=2))
    print('Saved three PDFs',flush=True)
if __name__=='__main__':main()
