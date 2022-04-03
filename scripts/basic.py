import numpy as np
from numba import njit

# Magnitude of vectors
@njit
def mag(vs):
    return np.sqrt(np.sum(np.square(vs), axis = 1))

# Linear combination of vectors with coefficients given by numbers
@njit
def nums_vs(nums, vs):
    return np.multiply(vs.T, nums).T 