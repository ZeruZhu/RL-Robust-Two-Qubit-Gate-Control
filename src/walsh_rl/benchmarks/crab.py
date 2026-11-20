"""Continuous-adjoint CRAB optimization with L-BFGS-B."""
from __future__ import annotations
import csv
import time
import numpy as np
from scipy.optimize import minimize
from .continuous_adjoint import fidelity_gradient, projection_pullback

def write_csv(path, rows):
    if not rows:
        return
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def validate_gradient(pulse, dynamics, settings, generators, nodes, weights, seed):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),8831,pulse.k]))
    theta=rng.normal(0,.0005,pulse.shape)
    theta[11:,0] -= .02*pulse.base[11:]/pulse.scales[11:]
    theta=theta.ravel()
    value,grad=fidelity_gradient(dynamics,pulse.project(theta),nodes,weights,generators)
    grad=projection_pullback(pulse,theta,grad,settings['projection_fd_step'])
    checks=[]
    for _ in range(3):
        direction=rng.normal(size=pulse.dimension);direction/=np.linalg.norm(direction)
        eps=2e-5
        plus=dynamics.objective(pulse.project(theta+eps*direction),nodes,weights)[0]
        minus=dynamics.objective(pulse.project(theta-eps*direction),nodes,weights)[0]
        fd=(plus-minus)/(2*eps);adj=float(grad@direction)
        checks.append(dict(finite_difference=fd,adjoint=adj,absolute_error=abs(fd-adj)))
    direct=dynamics.objective(pulse.project(theta),nodes,weights)[0]
    report=dict(directional_checks=checks,forward_difference=abs(value-direct))
    if max(x['absolute_error'] for x in checks)>2e-5 or report['forward_difference']>2e-7:
        raise AssertionError(f'Gradient validation failed: {report}')
    return report


def optimize(pulse, dynamics, settings, generators, nodes, weights, trace_path):
    """Held-out noise is deliberately absent from the optimizer interface."""
    best=dict(value=-np.inf,theta=None)
    trace=[];calls=0
    started=time.perf_counter();cpu_started=time.process_time()
    def record(_):
        trace.append(dict(iteration=len(trace),objective_average_gate_fidelity=best['value'],
                          objective_gradient_calls=calls,optimizer_cpu_seconds=time.process_time()-cpu_started,
                          optimizer_seconds=time.perf_counter()-started))
        write_csv(trace_path,trace)
    def objective(theta):
        nonlocal calls
        value,grad=fidelity_gradient(dynamics,pulse.project(theta),nodes,weights,generators)
        grad=projection_pullback(pulse,theta,grad,settings['projection_fd_step'])
        if not np.isfinite(value) or not np.all(np.isfinite(grad)) or not .2-1e-7 <= value <= 1+1e-7:
            raise FloatingPointError('Invalid robust objective or gradient')
        calls+=1
        if value>best['value']:
            best.update(value=value,theta=np.array(theta).copy())
        if calls==1:
            record(None)
        return 1-value,-grad
    result=minimize(objective,np.zeros(pulse.dimension),jac=True,method='L-BFGS-B',callback=record,
                    options=dict(maxiter=settings['iterations'],maxls=settings['maxls'],maxcor=settings['maxcor'],
                                 ftol=settings['ftol'],gtol=settings['gtol'],maxfun=settings['maxfun']))
    if best['theta'] is None:
        raise RuntimeError('No valid pulse evaluated')
    assert np.all(np.diff([r['objective_average_gate_fidelity'] for r in trace])>=0)
    info=dict(scipy_status=int(result.status),scipy_message=str(result.message),converged=bool(result.success),
              completed_updates=int(result.nit),objective_gradient_calls=calls,
              optimizer_cpu_seconds=time.process_time()-cpu_started,optimizer_seconds=time.perf_counter()-started,
              initial_objective_average_gate_fidelity=trace[0]['objective_average_gate_fidelity'])
    return best,info
