import random
from typing import Callable, Generator

import numpy as np
from numba import njit

from scripts.basic import randdir


# Proposal function for Metropolis–Hastings algorithm, using here a gaussian with width sigma
@njit
def proposal(x: float, sigma: float = 0.01) -> float:
    return np.random.normal(x, sigma)

# Metropolis–Hastings algorithm for sampling from a probability distribution
@njit
def metropolis(AC: object, pdistr: Callable, n: int, x0: float = 0.1, sigma: float = 0.05) -> Generator:
    x = x0 # start somewhere
    for _ in range(n):
        trial = proposal(x, sigma=sigma) # random neighbor from the proposal distribution
        acceptance = pdistr(AC, trial)/pdistr(AC, x)[0]
        
        # accept the move conditionally
        if random.random() < acceptance:
            x = trial

        yield x*randdir()

# Radial probability distribution for an axion cluster
@njit
def rdistr(AC: object, x: float) -> float:
    r = np.empty((1,3))
    r[0][0] = x
    r[0][1], r[0][2] = 0., 0.
        
    return AC.rtrunc() * x**2 * AC.rho_prf(AC.rCM + AC.rtrunc() * r)
    

