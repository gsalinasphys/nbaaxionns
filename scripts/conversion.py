from math import cos, pi, sin
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq, root_scalar

from scripts.basic import mag, mydot, randdirs3d, zeroat
from scripts.globals import ma, outdir
from scripts.orbits import smoothtraj


# Estimated conversion radius
def rc(NS: object, position: np.ndarray, time: float, exact: bool = False) -> float:
    dir = position/mag(position)
    if not exact:
        def to_min(x: float) -> float:
            return NS.wp(x*dir, time) - ma
        
    return root_scalar(to_min, bracket=[1e-4*NS.radius, 100*NS.radius], xtol=1e-10, rtol=1e-10).root

# Points on conversion surface
def conv_surf(NS: object, time: float, nsamples: int = 100_000, exact: bool = False) -> Generator:    
    for dir in randdirs3d(nsamples):
        rcii = rc(NS, dir, time, exact=exact)
        if rcii:
            yield rcii*dir

# Find positions at which trajectory crosses neutron star conversion surface, input a time ordered trajectory
def hits(NS: object, traj: np.ndarray, pprecision: int = 10_000) -> np.ndarray:
    rvin = traj[0][1:-1]
    traj = traj[1:]
    smthtraj = smoothtraj(traj)
    tmin, tmax = min(smthtraj[0]), max(smthtraj[0])
    ts = np.linspace(0., tmax-tmin, pprecision)
        
    def toroot(t):
        try:
            return mag(smthtraj[1](tmin+t).T) - rc(NS=NS, position=smthtraj[1](tmin+t).T, time=tmin+t)
        except ValueError as e:
            print("Error in finding hits (func call): ", e)
            return None
            
    zeros = zeroat(np.array([toroot(t) for t in ts if toroot(t)]))

    sols = []
    for zero in zeros:
        try:
            tsol = brentq(toroot, a=ts[zero], b=ts[zero+1], xtol=1e-16)
            xsol, ysol, zsol = smthtraj[1](tmin+tsol)
            vxsol, vysol, vzsol = smthtraj[2](tmin+tsol)
            sols.append([tmin+tsol, xsol, ysol, zsol, vxsol, vysol, vzsol, traj.T[-1][0], rvin[0], rvin[1], rvin[2], rvin[3], rvin[4], rvin[5]])
        except (ValueError, SystemError) as e:
            print("Error in finding hits (root finding): ", e)
            continue
    
    return sols

def allhits(NS: object, trajs: np.ndarray, pprecision: int = 10_000) -> np.ndarray:
    for ii, traj in enumerate(trajs):
        yield [[ii] + hit for hit in hits(NS, traj, pprecision)]
        
def plot_hits(ahits: np.ndarray, eventname: str, nmax: int = 100_000) -> None:
    X, Y, Z = np.array(ahits)[:nmax].T[2:5]
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, s=0.1, c='purple', linewidths=0)
    ax.set_xlabel('$x$ (km)')
    ax.set_ylabel('$y$ (km)')
    ax.set_zlabel('$z$ (km)')

    plt.savefig(outdir + eventname + '/' + eventname + 'conversion.png')
    plt.close()
