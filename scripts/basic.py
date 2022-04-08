from typing import Generator

import numpy as np
from numba import njit


# Repeat a numpy array n times
@njit
def repeat(v: np.ndarray, n: int) -> np.ndarray:
    nvs = np.empty((n, len(v)))
    for ii in range(n):
        nvs[ii] = v
    return nvs

# Magnitude of vectors
@njit
def mag(vs: np.ndarray) -> np.ndarray:
    if vs.ndim == 1:
        return np.sqrt(np.sum(vs**2))
    
    return np.sqrt(np.sum(vs**2, axis = 1))

# Dot product, faster than Numpy's np.dot for smaller vectors
@njit
def mydot(v1: np.ndarray, v2: np.ndarray) -> float:
    prods = np.empty_like(v1)
    for ii in range(len(v1)):
        prods[ii] = v1[ii]*v2[ii]
        
    return sum(prods)
        
# Linear combination of vectors with coefficients given by numbers
@njit
def nums_vs(nums: np.ndarray, vs: np.ndarray) -> np.ndarray:
    return np.multiply(vs.T, nums).T 
    
# Heaviside function
@njit
def heav(v: np.ndarray, x0: float) -> np.ndarray:
    if isinstance(v, float):
        if v > 0:
            return 1.
        elif v < 0:
            return 0.
        else:
            return x0
             
    bools = np.empty_like(v)
    for ii in range(len(v)):
        if v[ii] > 0:
            bools[ii] = 1.
        elif v[ii] < 0:
            bools[ii] = 0.
        else:
            bools[ii] = x0
            
    return bools

# Step function componentwise
@njit
def step(v: np.ndarray, xbounds: tuple = (0., 1.)):
    return heav(xbounds[1] - v, 1.) - heav(xbounds[0] - v, 0.)

# Given cond>0 return outide else inside componentwise
@njit
def cases(cond: np.ndarray, inside: np.ndarray, outside: np.ndarray):
    return inside*heav(-cond, 0.) + outside*heav(cond, 1.)

# Random directions in 2d centered at and angle with spread delta
@njit
def randdirs2d(n: int, center: float = 0., delta: float = np.pi) -> np.ndarray:
    thetas = np.random.uniform(center-delta, center+delta, n)
    return np.vstack((np.cos(thetas), np.sin(thetas))).T

# Method to draw uniformly distributed points along the unit sphere (Marsaglia 1972)
@njit
def randdir3d() -> np.ndarray:
    found = False
    while not found:
        x1, x2 = np.random.uniform(-1., 1., 2)
        xnorm = x1**2 + x2**2
        
        if xnorm < 1:
            found = True

    nx = 2*x1*np.sqrt(1 - xnorm)
    ny = 2*x2*np.sqrt(1 - xnorm)
    nz = 1 - 2*xnorm

    return np.array([nx, ny, nz])

@njit(fastmath=True)
def randdirs3d(n: int) -> Generator:
    for ii in range(n):
        yield randdir3d()
