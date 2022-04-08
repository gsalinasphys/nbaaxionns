from typing import Callable

import numpy as np
from classes.ns import NeutronStar
from classes.particles import Particles
from numba import njit

from scripts.basic import mag, repeat
from scripts.metropolis import metropolis, rdistr
from scripts.orbits import add_ps, min_approach, rm_far, trajs, update_ps


# Roche radius (km), NS mass in M_Sun and axion clump mass in 10^{-10}M_Sun
@njit(cache=True)
def roche(AC: object, NS: object) -> float:
    return AC.rtrunc()*np.power(2*NS.mass/(1e-10*AC.mass), 1/3)

# Full trajectories, use batches of 100 particles for max speed
# !!!!! Jitted version is slower
def trajAC(pAC: object, NS: object, rmin: float = None, rprecision: float = 1e-2) -> None:
    if min_approach(pAC, NS) > rmin:
        return None
    
    finished = False  
    while not finished:
        if mag(pAC.positions[0]) < rmin:
            finished = True
        
        update_ps(pAC, NS, rprecision=rprecision)

    return pAC.impact_param()[0]    # Return impact parameter

def selectrvs(AC: object, NS: object, nps: int, rmax: float, rcyl: float, b: float, nsamples: int = 10_000_000) -> np.ndarray:
    ps = Particles(np.empty((0,1)), np.empty((0,1)))
    while len(ps.positions) < nps:
        rs = metropolis(rdistr, nsamples, x0=b/AC.rtrunc(), sigma=1e-1*rcyl/AC.rtrunc(), rbounds=(b-rcyl, b+rcyl))
        rsdrawn = AC.rCM + np.array(list(rs))
        rsdrawn = rsdrawn[mag(rsdrawn[:, [0, 2]]) < rcyl]
        
        vsdrawn = repeat(AC.vCM, len(rsdrawn))
        if AC.vdisptype:
            vsdrawn += np.array(list(AC.vsdisp(rsdrawn)))
        accelerations, times = repeat(np.zeros(3), len(rsdrawn)), np.repeat(0., len(rsdrawn))
        add_ps(ps, rsdrawn, vsdrawn, accelerations, times)
    
        rm_far(ps, NS, rmax)
        print(len(ps.positions))
        
    return (ps.positions, ps.velocities)
            
# Run trajectories of particles that will get close enough to the Neutron star
def runtrajs(p: object, NS: object, rmax: float, rprecision: float = 1e-2) -> np.ndarray:
    return trajs(p, NS, rlimits=(NS.radius, rmax), retval=True, rprecision=rprecision)

# Top version of the function above to run with pool.starmap
def truntrajs(runtrajs: Callable, rvsin: tuple, rmax: float, NSparams: tuple = (1.,10.,1.,np.array([0.,0.,1.]),1.,0.,0.), rprecision: float = 1e-2) -> np.ndarray:
    ps = Particles(rvsin[0], rvsin[1])
    mass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=mass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    return runtrajs(ps, NS, rmax, rprecision=rprecision)

# def trajps(AC: object, NS: object, nps:)
