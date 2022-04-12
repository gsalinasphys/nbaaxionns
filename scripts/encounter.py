from math import sqrt
from typing import Callable

import numpy as np
from classes.ns import NeutronStar
from classes.particles import Particles
from numba import njit

from scripts.basic import mag, randdir2d, repeat
from scripts.globals import G
from scripts.metropolis import metropolis, rincyl
from scripts.orbits import add_ps, min_approach, rm_far, trajs, update_ps


# Roche radius (km), NS mass in M_Sun and axion clump mass in 10^{-10}M_Sun
@njit
def roche(AC: object, NS: object) -> float:
    return AC.rtrunc()*np.power(2*NS.mass/(1e-10*AC.mass), 1/3)

# Full trajectories, use batches of 100 particles for max speed
def trajAC(pAC: object, NS: object, rmin: float = None, rprecision: float = 5e-2) -> None:
    if min_approach(pAC, NS) > rmin:
        return None
    
    finished = False  
    while not finished:
        if mag(pAC.positions[0]) < rmin:
            finished = True
        
        update_ps(pAC, NS, rprecision=rprecision)

# def bmax(AC: object, NS: object, rmax: float) -> float:
#     return rmax**2 + 2*G*NS.mass*rmax/mag(AC.vCM)**2

# Safe cylinder radius to draw particles from
def cylmax(AC: object, NS: object):
    rtry = AC.deltav()*roche(AC, NS) / (mag(AC.vCM)*AC.rtrunc())
    if rtry < AC.rtrunc():
        return rtry
    
    return AC.rtrunc()

def selectrvs(AC: object, NS: object, nps: int, rmax: float, cylbnds: tuple = ((0., 1.), (0., 1.)), nsamples: int = 1_000_000) -> np.ndarray:
    zcyl, L = 0.5*cylbnds[1][0]+0.5*cylbnds[1][1], cylbnds[1][1]-cylbnds[1][0]
    rcyl, drcyl = 0.5*cylbnds[0][0]+0.5*cylbnds[0][1], cylbnds[0][1]-cylbnds[0][0]
    
    ps = Particles(np.empty((0,1)), np.empty((0,1)))
    while len(ps.positions) < nps:
        rs = metropolis(rincyl, nsamples, x0=np.append(rcyl*randdir2d(), zcyl),
                        sigma=np.array([drcyl/10., drcyl/10., 0.1*L]),
                        xbounds=cylbnds, O=AC)
        rsdrawn = AC.rtrunc()*np.array(list(rs))
        
        vsdrawn = repeat(AC.vCM, len(rsdrawn))
        if AC.vdisptype:
            # vsdrawn += np.array(list(AC.vsdisp(rsdrawn, 0.1*bmax(AC, NS, rmax))))
            vsdrawn += np.array(list(AC.vsdisp(rsdrawn)))
        accelerations, times = repeat(np.zeros(3), len(rsdrawn)), np.repeat(0., len(rsdrawn))
        
        add_ps(ps, rsdrawn, vsdrawn, accelerations, times)
        rm_far(ps, NS, rmax)
        print(len(ps.positions))
        
    return (ps.positions, ps.velocities)
            
# Run trajectories of particles that will get close enough to the Neutron star
def runtrajs(p: object, NS: object, rmax: float, rprecision: float = 5e-2) -> np.ndarray:
    return trajs(p, NS, rlimits=(NS.radius, rmax), retval=True, rprecision=rprecision)

# Top version of the function above to run with pool.starmap
def truntrajs(runtrajs: Callable, rvsin: tuple, rmax: float, NSparams: tuple = (1.,10.,1.,np.array([0.,0.,1.]),1.,0.,0.), rprecision: float = 5e-2) -> np.ndarray:
    ps = Particles(rvsin[0], rvsin[1])
    mass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=mass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    return runtrajs(ps, NS, rmax, rprecision=rprecision)

