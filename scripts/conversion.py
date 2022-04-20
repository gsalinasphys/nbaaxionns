from math import cos, pi, sin
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import brentq, root_scalar

from scripts.basic import mag, mydot, nums_vs, randdirs3d, repeat, zeroat
from scripts.globals import G_eV2, c, eV_GHz, gag, km_eVinv, maGHz, outdir
from scripts.orbits import smoothtraj


# Estimated conversion radius
def rc(NS: object, position: np.ndarray, time: float, exact: bool = False) -> float:
    dir = position/mag(position)
    if not exact:
        def to_min(x: float) -> float:
            return NS.wp(x*dir, time) - maGHz
        
    try:
        rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius], xtol=1e-10, rtol=1e-10).root
        if rc > NS.radius and rc < 100*NS.radius:
            return rc
    except ValueError as e:
        # print("Error in rc determination (root finding): ", e)
        return None

# Points on conversion surface
def conv_surf(NS: object, time: float, nsamples: int = 100_000, exact: bool = False) -> Generator:    
    for dir in randdirs3d(nsamples):
        rcii = rc(NS, dir, time, exact=exact)
        if rcii:
            yield rcii*dir

# Find positions at which trajectory crosses neutron star conversion surface
def hits(NS: object, traj: np.ndarray, pprecision: int = 100) -> np.ndarray:
    smthtraj = smoothtraj(traj, NS.T)
    tmin, tmax = min(traj.T[0]), max(traj.T[0])
    ts = np.linspace(0., tmax-tmin, pprecision)
        
    def toroot(t):
        try:
            return mag(smthtraj[0](tmin+t).T)/rc(NS=NS, position=smthtraj[0](tmin+t).T, time=tmin+t) - 1
        except (TypeError, ValueError, SystemError) as e:
            # print("Error in finding hits (function call): ", e)
            return None
    
    zeros = zeroat(np.array([toroot(t) for t in ts if toroot(t)]))
    
    sols = []
    for zero in zeros:
        try:
            tsol = brentq(toroot, a=ts[zero], b=ts[zero+1], xtol=1e-16)
            xsol, ysol, zsol = smthtraj[0](tmin+tsol)
            vxsol, vysol, vzsol = smthtraj[1](tmin+tsol)
            sols.append([tsol, xsol, ysol, zsol, vxsol, vysol, vzsol, traj.T[-1][0]])
        except (ValueError, SystemError) as e:
            # print("Error in finding hits (root finding): ", e)
            continue
        
    return sols

def allhits(NS: object, trajs: np.ndarray, pprecision: int = 500) -> np.ndarray:
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

def prob(hit: np.ndarray, NS: object) -> float:
    time, position, velocity = hit[1], hit[2:5], hit[5:8]

    k = maGHz*mag(velocity)/c    # in GHz
    theta = mydot(NS.B(position, time), velocity)/(mag(NS.B(position, time))*mag(velocity))
    
    ydir = np.cross(velocity, np.cross(velocity, NS.B(position, time)))
    ydir /= mag(ydir)
    sdir = cos(theta)*ydir + sin(theta)*velocity/mag(velocity)
    sdir /= mag(sdir)

    eps = 1e-8
    dr = sdir*eps
    dwp = NS.wp(position+dr, time) - NS.wp(position, time)
    
    wpp = dwp/eps
    
    return eV_GHz * G_eV2**2 * km_eVinv * 1.e-18/2. * (gag*mag(NS.B(position, time))*sin(theta))**2 * pi * maGHz**5 / (2*k*abs(wpp)) / (k**2+(maGHz*sin(theta))**2)**2
