"""Validate bundled/fresh results and compare numerical errors to the figure inputs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
METHODS={'naive_mean','naive_median','srf_noalignment','SF','SRF'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mnist-results',type=Path,default=ROOT/'mnist/reference_results')
    p.add_argument('--simulation-results',type=Path,default=ROOT/'simulation/reference_results')
    p.add_argument('--output',type=Path,default=ROOT/'validation_outputs/numerical_verification.json')
    args=p.parse_args();md=args.mnist_results;sd=args.simulation_results
    df=pd.read_csv(md/'metrics.csv');assert len(df)==150 and set(df.method)==METHODS
    assert not df.duplicated(['run','sigma','method']).any()
    assert np.isfinite(df.recovery_error).all() and df.recovery_error.between(0,np.sqrt(2)+1e-12).all()
    assert (df.groupby(['sigma','method']).size()==5).all()
    ref=np.load(md/'reference.npz')['atoms'];assert ref.shape==(784,10)
    np.testing.assert_allclose(ref.T@ref,np.eye(10),atol=1e-10)
    verified=0
    for sigma in [0.,2.]:
        with np.load(md/'run_0'/f'sigma{sigma:.3f}.npz') as d:
            for m in METHODS:
                a=d[m+'_aligned'];assert a.shape==ref.shape
                np.testing.assert_allclose(np.linalg.norm(a,axis=0),1,atol=1e-10)
                e=np.linalg.norm(a-ref)/np.sqrt(10)
                target=df[df.run.eq(0)&df.sigma.eq(sigma)&df.method.eq(m)].recovery_error.iloc[0]
                np.testing.assert_allclose(e,target,atol=1e-10);verified+=1
    baseline=pd.read_csv(ROOT/'mnist/reference_results/metrics.csv')
    merged=df.merge(baseline,on=['run','sigma','method'],suffixes=('_run','_ref'));assert len(merged)==150
    mdiff=float(abs(merged.recovery_error_run-merged.recovery_error_ref).max())
    config=np.load(ROOT/'mnist/config/partitions.npz')
    for run in range(5):
        a=config[f'indices_{run}'];assert a.shape==(10,6310) and len(np.unique(a))==63100
        np.testing.assert_array_equal(np.sort(a.ravel()),np.sort(config['central_indices']))
    result=loadmat(sd/'simulation_results.mat',simplify_cells=True)['results']
    reference=loadmat(ROOT/'simulation/reference_results/simulation_results.mat',simplify_cells=True)['results']
    assert result['repeats']==10
    maximum=0.;num=0
    for exp,old in zip(result['all_results'],reference['all_results']):
        assert exp['exp_mode']==old['exp_mode'];np.testing.assert_array_equal(exp['sweep'],old['sweep'])
        a=exp['errors_repeats'];b=old['errors_repeats'];assert a.shape==b.shape
        assert np.isfinite(a).all() and (a>=0).all()
        np.testing.assert_allclose(a.mean(0),exp['errors'],atol=1e-12)
        np.testing.assert_allclose(a.std(0,ddof=1),exp['errors_std'],atol=1e-12)
        maximum=max(maximum,float(abs(a-b).max()));num+=a.size
    assert num==1000
    provenance=json.loads((ROOT/'provenance.json').read_text())
    for fig in provenance['figures']:
        assert hashlib.sha256((ROOT/fig['snapshot']).read_bytes()).hexdigest()==fig['sha256']
    report=dict(mnist_evaluations=150,mnist_display_dictionaries_verified=verified,
        mnist_max_error_difference_from_archive=mdiff,simulation_evaluations=num,
        simulation_max_error_difference_from_archive=maximum,original_pdf_hashes_verified=5,
        matches_archive_within_tolerance=bool(mdiff<1e-7 and maximum<1e-7))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['matches_archive_within_tolerance']:
        raise SystemExit('Fresh results differ from archive; inspect the saved report. No results were replaced.')
if __name__=='__main__':main()
