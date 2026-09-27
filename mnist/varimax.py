"""Original simultaneous SVD Varimax updates; no Kaiser normalization."""
import numpy as np

def orthogonal(rng, k):
    q, r = np.linalg.qr(rng.normal(size=(k,k)))
    return q * np.where(np.diag(r)<0, -1, 1)

def objective(x, r):
    y=x@r; y2=y*y
    return float(np.mean(y2*y2, axis=0).sum() - (np.mean(y2,axis=0)**2).sum())

def varimax(x, seed, max_iter=6000, tol=1e-8, initial=None):
    # One global rescaling preserves the optimizer; no row normalization.
    scale=np.sqrt(np.mean(x*x)); z=x/scale; n,k=z.shape
    r=(np.eye(k) if seed == -1 else orthogonal(np.random.default_rng(seed),k)) if initial is None else initial.copy()
    previous=objective(z,r); min_gain=0.; converged=False
    for iteration in range(1,max_iter+1):
        y=z@r; y2=y*y
        b=z.T@(y*y2-y*np.mean(y2,axis=0)) / n
        u,_,vt=np.linalg.svd(b,full_matrices=False); new=u@vt
        delta=float(np.linalg.norm(new-r,'fro')/np.sqrt(k))
        r=new
        value=objective(z,r)
        min_gain=min(min_gain,(value-previous)/max(abs(previous),1.))
        previous=value
        if delta<tol:
            converged=True; break
    y=z@r; y2=y*y; g=4*z.T@(y*y2-y*np.mean(y2,axis=0))/n
    rg=r.T@g; residual=np.linalg.norm(rg-rg.T)/max(np.linalg.norm(g),1e-15)
    np.testing.assert_allclose(r.T@r,np.eye(k),atol=2e-12)
    if min_gain < -1e-8: raise RuntimeError('Varimax objective decreased materially')
    return r,dict(seed=seed,iterations=iteration,converged=converged,rotation_step=delta,
        objective=objective(x,r),scaled_objective=value,stationarity_residual=float(residual),minimum_relative_gain=min_gain)
