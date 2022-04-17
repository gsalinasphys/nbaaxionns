import random
import string
from math import cos, pi, sin, sqrt
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
    if isinstance(vs, float):
        return abs(vs)
    elif vs.ndim == 1:
        return sqrt(np.sum(vs**2))
    
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

# Find index of element in array closest to a value 
@njit
def nearest(arr: np.ndarray, val: float) -> int:
    return (np.abs(arr - val)).argmin()

@njit
def nearests(arr: np.ndarray, vals: np.ndarray) -> int:
    for ii in range(len(vals)):
        yield nearest(arr, vals[ii])

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

# Random direction in 2d centered at and angle with spread delta
@njit
def randdir2d(center: float = 0., delta: float = pi) -> np.ndarray:
    delta = min(delta, pi)
    theta = random.uniform(center-delta, center+delta)
    return np.array([cos(theta), sin(theta)])

@njit
def randdirs2d(n: int, center: float = 0., delta: float = pi) -> Generator:
    for ii in range(n):
        yield randdir2d(center, delta)
    
# Method to draw uniformly distributed points along the unit sphere (Marsaglia 1972)
@njit
def randdir3d() -> np.ndarray:
    found = False
    while not found:
        x1, x2 = np.random.uniform(-1., 1., 2)
        xnorm = x1**2 + x2**2
        
        if xnorm < 1:
            found = True

    nx = 2*x1*sqrt(1 - xnorm)
    ny = 2*x2*sqrt(1 - xnorm)
    nz = 1 - 2*xnorm

    return np.array([nx, ny, nz])

@njit(fastmath=True)
def randdirs3d(n: int) -> Generator:
    for ii in range(n):
        yield randdir3d()
        
# Find where array crosses zero
@njit
def zeroat(v: np.ndarray) -> np.ndarray:
    prods = v[1:]*v[:-1]
    return np.where(prods < 0)[0]
