import random
from math import atan, pi
from typing import Callable, Generator

import numpy as np
from numba import njit

from scripts.basic import mag, step
from scripts.globals import EmptyClass


# Proposal function for Metropolis–Hastings algorithm, using here a gaussian with width sigma
@njit
def proposal(x: float, sigma: float = 0.1) -> float:
    if isinstance(x, float):
        return np.random.normal(x, sigma)
    
    toret = np.empty_like(x)
    for ii in range(len(x)):
        toret[ii] = np.random.normal(x[ii], sigma[ii])
        
    return toret

# Metropolis–Hastings algorithm for sampling from a probability distribution
@njit
def metropolis(pdistr: Callable, n: int, x0: float = 0.5, sigma: float = 0.1, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> Generator:
    x = x0 # start somewhere
    for _ in range(n):
        trial = proposal(x, sigma=sigma) # random neighbor from the proposal distribution
        acceptance = pdistr(trial, xbounds, O)/pdistr(x, xbounds, O)
        
        # accept the move conditionally
        if random.random() < acceptance:
            x = trial

        yield x

# Step distribution in any number of dimensions
@njit
def stepdistr(v: np.ndarray, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> float:
    if isinstance(v, float):
        return step(v, xbounds)
    
    toret = 1.
    for ii in range(len(v)):
        toret = toret*step(v[ii], xbounds)
        
    return toret

# Radial probability distribution for an axion clump
@njit
def rdistr(r: float, rbounds: tuple = (0., 1.), AC: object = EmptyClass()) -> float:
    if isinstance(r, float):
        return r**2 * AC.rho_prf(AC.rtrunc()*r, rbounds)
    
    return AC.rho_prf(AC.rtrunc()*r+AC.rCM, rbounds)

# Step distribution for 2d and 3d (or any higher d), for 1d use step instead
@njit
def cyldistr(v: np.ndarray, cylbounds: tuple = ((0., 1.), (0., 1.)), O: object = EmptyClass()) -> float:    # Tuple cylbounds in the format ((0., rcyl), (Lmin, Lmax))
    return step(mag(v[:2]), cylbounds[0]) * step(v[2], cylbounds[1])   # Axis of cylinder along z-axis

# Distribution for axion clump points inside a cylinder with axis along the z-axis
@njit
def rincyl(r: np.ndarray, cylbounds: tuple = ((0., 1.), (0., 1.)), AC: object = EmptyClass()) -> float:
    return rdistr(r-AC.rCM*np.array([1.,0.,0.])/AC.rtrunc(), (0., 1.), AC) * cyldistr(r, cylbounds, AC)

