"""Evaluation-only signed permutation matching against a frozen reference."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def normalize(a):
    a=np.asarray(a,dtype=float)
    norms=np.linalg.norm(a,axis=0)
    if not np.isfinite(a).all() or np.any(norms<1e-12):
        raise ValueError('A dictionary contains nonfinite or numerically zero atoms')
    return a/norms

def compare(estimate,reference):
    a=normalize(estimate);d=normalize(reference)
    if a.shape!=d.shape:raise ValueError('Dictionary shapes differ')
    correlations=d.T@a
    rows,cols=linear_sum_assignment(-np.abs(correlations))
    assert np.array_equal(rows,np.arange(d.shape[1]))
    signs=np.where(correlations[rows,cols]<0,-1,1)
    aligned=a[:,cols]*signs
    cos=np.clip(np.sum(d*aligned,axis=0),0,1)
    distances=np.linalg.norm(aligned-d,axis=0)
    error=float(np.linalg.norm(aligned-d,'fro')/np.sqrt(d.shape[1]))
    np.testing.assert_allclose(error**2,2*(1-cos.mean()),atol=1e-12)
    ud,sd,_=np.linalg.svd(d,full_matrices=False)
    ua,sa,_=np.linalg.svd(a,full_matrices=False)
    rank_d=int(np.sum(sd>sd[0]*max(d.shape)*np.finfo(float).eps))
    rank_a=int(np.sum(sa>sa[0]*max(a.shape)*np.finfo(float).eps))
    overlap=np.linalg.norm(ud[:,:rank_d].T@ua[:,:rank_a],'fro')**2
    # Normalized distance between the two orthogonal projection matrices; still
    # records missing directions if the estimated dictionary has lost rank.
    subspace=float(np.sqrt(max(rank_d+rank_a-2*overlap,0)/(2*rank_d)))
    summary=dict(recovery_error=error,mean_abs_cosine=float(cos.mean()),mean_angle_degrees=float(np.degrees(np.arccos(cos)).mean()),subspace_error=subspace,estimated_rank=rank_a,reference_rank=rank_d)
    match=dict(permutation=cols,signs=signs,cosines=cos,atom_errors=distances,aligned_atoms=aligned)
    return summary,match
