from typing import Generator

import numpy as np
from scipy.optimize import brentq, root_scalar

from scripts.basic import mag, nums_vs, randdirs3d, repeat, zeroat
from scripts.globals import maGHz
from scripts.orbits import smoothtraj


def rc(NS: object, position: np.ndarray, time: float, exact: bool = False) -> float:    # Estimated conversion radius in some direction
    dir = position/mag(position)
    if not exact:
        def to_min(x: float) -> float:
            return NS.wp(x*dir, time) - maGHz
        
    try:
        rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius], xtol=1e-10, rtol=1e-10).root
        if rc > NS.radius and rc < 100*NS.radius:
            return rc
    except ValueError:
        return None

def conv_surf(NS: object, time: float, nsamples: int = 10_000, exact: bool = False) -> Generator:    
    for dir in randdirs3d(nsamples):
        rcii = rc(NS, dir, time, exact=exact)
        if rcii is not None:
            yield rcii*dir

# Find positions at which trajectory crosses neutron star conversion surface
def hits(NS: object, traj: np.ndarray, pprecision: int = 100) -> np.ndarray:
    smthtraj = smoothtraj(traj)
    tmin, tmax = min(traj.T[0]), max(traj.T[0])
    ts = np.linspace(0., tmax-tmin, pprecision)
        
    def toroot(t):
        try:
            return mag(smthtraj[0](tmin+t).T)/rc(NS=NS, position=smthtraj[0](tmin+t).T, time=tmin+t) - 1
        except TypeError:
            return None
    
    zeros = zeroat(np.array([toroot(t) for t in ts if toroot(t) is not None]))
    
    tsols = []
    for zero in zeros:
        try:
            tsol = brentq(toroot, a=ts[zero], b=ts[zero+1], xtol=1e-16)
            tsols.append([tsol, smthtraj[0](tmin+tsol), smthtraj[1](tmin+tsol)])
        except ValueError:
            continue
        
    return tsols

def allhits(NS: object, trajs: np.ndarray, pprecision: int = 100) -> np.ndarray:
    for traj in trajs:
        yield hits(NS, traj, pprecision)
