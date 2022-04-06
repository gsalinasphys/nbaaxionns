import numpy as np
from numba import njit


# Magnitude of vectors
@njit
def mag(vs: np.ndarray) -> np.ndarray:
    if vs.ndim == 1:
        return np.sqrt(np.sum(vs**2))
    
    return np.sqrt(np.sum(vs**2, axis = 1))

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

@njit
def rm_inds(arr: np.ndarray, inds: np.ndarray) -> np.ndarray:
    new_arr, rmvd = np.empty_like(arr), 0
    for ii in range(len(arr)):
        if ii not in inds:
            new_arr[ii-rmvd] = arr[ii]
        else:
            rmvd += 1
            
    return new_arr[:-rmvd]
    
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

# Method to draw uniformly distributed points along the unit sphere (Marsaglia 1972)
@njit
def randdir() -> np.ndarray:
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
def randdirs(n: int) -> np.ndarray:
    for ii in range(n):
        yield randdir()
