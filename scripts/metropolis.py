import random
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
def metropolis(pdistr: Callable, n: int, x0: float = 0.1, sigma: float = 0.1, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> Generator:
    x = x0 # start somewhere
    for _ in range(n):
        trial = proposal(x, sigma=sigma) # random neighbor from the proposal distribution
        acceptance = pdistr(trial, xbounds, O)/pdistr(x, xbounds, O)
        
        # accept the move conditionally
        if random.random() < acceptance:
            x = trial

        yield x

# Radial probability distribution for an axion cluster
@njit
def rdistr(r: float, rbounds: tuple = (0., 1.), AC: object = EmptyClass()) -> float:
    if isinstance(r, float):
        return r**2 * AC.rho_prf(AC.rtrunc() * r, rbounds)
    
    return AC.rho_prf(AC.rtrunc() * r, rbounds)

# Step distribution in any number of dimensions
@njit
def stepdistr(v: np.ndarray, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> float:
    if isinstance(v, float):
        return step(v, xbounds)
    
    toret = 1.
    for ii in range(len(v)):
        toret = toret*step(v[ii], xbounds)
        
    return toret

# Step distribution for 2d and 3d (or any higher d), for 1d use step instead
@njit
def cyldistr(v: np.ndarray, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> float:
    return step(mag(v*np.array([1., 0., 1.])), (0., 1.)) * step(v[1], xbounds)   # Axis of cylinder along yaxis

@njit
def rincyl(r: np.ndarray, xbounds: tuple = (0., 1.), O: object = EmptyClass()) -> float:
    return rdistr(r, xbounds)
    