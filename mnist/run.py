"""Full, self-contained rerun of the pixel-side Varimax MNIST paper figures.

This deliberately reproduces A=U R, NOT the later sample-side U S R study.
No fitted dictionaries from other experiments are required.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from data import load_mnist
from varimax import varimax
from aggregation import prepare_srf_clusters,aggregate_srf_clusters,aggregate_naive,normalize_columns,geometric_median
from dictionary_metrics import compare
ROOT=Path(__file__).resolve().parent
METHODS=['naive_mean','naive_median','srf_noalignment','SF','SRF']
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def seed(*ids):return int(np.random.SeedSequence([20260912,101,*ids]).generate_state(1)[0])
def gm_columns(raw,labels,log):
    return normalize_columns(np.column_stack([geometric_median(raw[:,labels==j].T,max_iter=30000,tol=1e-7,diagnostics=log) for j in range(10)]))
def fit(x,tag,config,out,resume):
    meta=config['fits'][tag];image_hash=digest(x);assert image_hash==meta['image_sha256'],tag
    target=out/'fits'/tag;npz=Path(str(target)+'.npz');js=Path(str(target)+'.json')
    if resume and npz.exists() and js.exists():
        saved=json.loads(js.read_text());assert saved['image_sha256']==image_hash
        assert [l['seed'] for l in saved['starts']]==meta['seeds']
        return np.load(npz)['atoms']
    # Retain the exact operation order from the original clean/noisy fit paths.
    clean=x.dtype==np.uint8
    mean=x.mean(0)/255 if clean else x.mean(0)
    cov=np.zeros((784,784))
    for pos in range(0,len(x),4096):
        block=x[pos:pos+4096].astype(float)/255-mean if clean else x[pos:pos+4096]-mean
        cov+=block.T@block/len(x)
    values,u=eigh(cov,subset_by_index=(773,783));values=values[::-1];u=u[:,::-1][:,:10]
    rotations=[];logs=[]
    for initial_seed in meta['seeds']:
        r,log=varimax(u,initial_seed);rotations.append(r);logs.append(log)
    good=[i for i,l in enumerate(logs) if l['converged']];assert good,tag
    best=max(good,key=lambda i:logs[i]['objective']);r=rotations[best];a=u@r
    np.testing.assert_allclose(a.T@a,np.eye(10),atol=3e-12)
    np.savez_compressed(npz,atoms=a,pca_basis=u,pca_variances=values[:10],rotation=r,mean=mean)
    js.write_text(json.dumps(dict(tag=tag,image_sha256=image_hash,starts=logs,best_index=best,selected_seed=logs[best]['seed'],n=len(x)),indent=2))
    print('FIT',tag,'converged',len(good),'/',len(logs),flush=True);return a
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--resume',action='store_true',help='Reuse only validated fits in this output directory')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);(out/'fits').mkdir(exist_ok=True)
    start=time.time();config=json.loads((ROOT/'config/experiment.json').read_text());full,y=load_mnist(args.data_dir)
    partitions=np.load(ROOT/'config/partitions.npz');pool=partitions['central_indices']
    central_raw=fit(full[pool],'central',config,out,args.resume)
    # Centralized-only display orientation: not used in local fitting or server clustering.
    display=np.load(ROOT/'config/reference_display.npz')['atoms']
    _,match=compare(central_raw,display);ref=match['aligned_atoms']
    np.savez_compressed(out/'reference.npz',atoms=ref,raw_atoms=central_raw,display_permutation=match['permutation'],display_signs=match['signs'])
    rows=[];factors=[];gmlogs=[]
    for run in range(config['runs']):
        folder=out/f'run_{run}';folder.mkdir(exist_ok=True)
        indices=partitions[f'indices_{run}'];bad=partitions[f'bad_{run}']
        assert indices.shape==(10,6310) and len(np.unique(indices))==63100 and len(bad)==4
        np.testing.assert_array_equal(np.sort(indices.ravel()),np.sort(pool))
        np.savez_compressed(folder/'partition.npz',indices=indices,bad_clients=bad)
        clean=[]
        for k in range(10):
            np.testing.assert_array_equal(np.bincount(y[indices[k]],minlength=10),np.full(10,631))
            clean.append(fit(full[indices[k]],f'run{run}_client{k}',config,out,args.resume))
        clean=np.stack(clean);np.savez_compressed(folder/'clean.npz',atoms=clean)
        bysigma={s:clean.copy() for s in config['sigmas']}
        for k in bad:
            k=int(k);x=full[indices[k]].astype(float)/255
            z=np.random.default_rng(seed(20,run,k)).normal(size=x.shape)
            for sigma in config['sigmas'][1:]:
                bysigma[sigma][k]=fit(x+sigma*z,f'run{run}_client{k}_sigma{sigma:.3f}',config,out,args.resume)
        for sigma,uploaded in bysigma.items():
            state={};raw,labels,aligned=prepare_srf_clusters(uploaded,10,seed(4,run),state)
            assert len(np.unique(labels))==10
            saved=dict(uploaded_atoms=uploaded,raw_atoms=raw,labels=labels,aligned_atoms=aligned,reference=ref,**state)
            for key,fn in [('naive_mean',lambda log:aggregate_naive(uploaded,'mean',log)),
                ('naive_median',lambda log:gm_columns(np.concatenate(uploaded,axis=1),np.tile(np.arange(10),10),log)),
                ('srf_noalignment',lambda log:gm_columns(raw,labels,log)),
                ('SF',lambda log:aggregate_srf_clusters(aligned,labels,10,'mean',log)),
                ('SRF',lambda log:gm_columns(aligned,labels,log))]:
                log=[];a=fn(log);stat,m=compare(a,ref)
                assert all(l['converged'] for l in log)
                rows.append(dict(run=run,sigma=sigma,method=key,**stat))
                gmlogs.extend(dict(run=run,sigma=sigma,method=key,**l) for l in log)
                saved[key]=a;saved[key+'_aligned']=m['aligned_atoms']
                for j,e in enumerate(m['atom_errors']):factors.append(dict(run=run,sigma=sigma,method=key,factor=j+1,error=e))
            np.savez_compressed(folder/f'sigma{sigma:.3f}.npz',**saved)
            print('EVAL',run,sigma,flush=True)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    df=pd.DataFrame(rows);assert len(df)==150
    df.groupby(['sigma','method']).agg(error_mean=('recovery_error','mean'),error_sd=('recovery_error','std')).reset_index().to_csv(out/'summary.csv',index=False)
    pd.DataFrame(factors).to_csv(out/'per_factor_metrics.csv',index=False)
    (out/'geometric_median_diagnostics.json').write_text(json.dumps(gmlogs,indent=2))
    (out/'run_info.json').write_text(json.dumps(dict(fits=151,initializations=4601,method_evaluations=150,elapsed_seconds=time.time()-start,
        estimator='Pixel-side Varimax on U; A=U R; no singular-value weighting',noise='Unclipped iid Gaussian image noise'),indent=2))
    print('DONE',out,'elapsed',time.time()-start,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
