from functools import cache
from math import pi, sqrt, tan
from typing import Callable

import numpy as np
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
@njit
def trajAC(pAC: object, NS: object, rmin: float, rprecision: float = 5e-2) -> None:
    if min_approach(pAC, NS) > rmin:
        return None
    
    finished = False  
    while not finished:
        if mag(pAC.positions[0]) < rmin:
            finished = True
        
        update_ps(pAC, NS, rprecision=rprecision)

@njit
def bmax(AC: object, NS: object) -> float:
    return sqrt(NS.rcmax()**2 + 2*G*NS.mass*NS.rcmax()/mag(AC.vCM)**2)

# Safe cylinder radius to draw particles from
@njit
def cylmax(AC: object, NS: object) -> float:
    if not AC.vdisptype:
        return bmax(AC, NS)/AC.rtrunc()
    
    rtry = AC.deltav()*roche(AC, NS) / (mag(AC.vCM)*AC.rtrunc())
    if rtry < AC.rtrunc():
        return max((rtry, bmax(AC, NS)/AC.rtrunc()))
    
    return AC.rtrunc()

# Select initial conditions that will hit close to the neutron star
def selectrvs(AC: object, NS: object, nps: int, cylbnds: tuple = (-1., 1.), nsamples: int = 1_000_000) -> np.ndarray:
    zcyl, L = 0.5*cylbnds[0]+0.5*cylbnds[1], cylbnds[1]-cylbnds[0]
    
    ps = Particles(np.empty((0,1)), np.empty((0,1)))
    while len(ps.positions) < nps:
        rs = metropolis(rincyl, nsamples, x0=np.append(0.1*cylmax(AC, NS)*randdir2d(), zcyl),
                        sigma=np.array([cylmax(AC, NS)/10., cylmax(AC, NS)/10., 0.1*L]),
                        xbounds=((0., cylmax(AC, NS)), cylbnds), O=AC)
        rsdrawn = AC.rCM*np.array([0.,0.,1.]) + AC.rtrunc()*np.array(list(rs))
        
        vsdrawn = repeat(AC.vCM, len(rsdrawn))
        if AC.vdisptype:
            vsdrawn += np.array(list(AC.vsdisp(rsdrawn)))
        accelerations, times, nperiods = repeat(np.zeros(3), len(rsdrawn)), np.repeat(0., len(rsdrawn)), np.repeat(0, len(rsdrawn))
        
        add_ps(ps, rsdrawn, vsdrawn, accelerations, times, nperiods)
        rm_far(ps, NS)
        
    return (ps.positions, ps.velocities)
