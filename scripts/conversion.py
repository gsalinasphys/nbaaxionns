import numpy as np
from numba import njit
from scipy.optimize import root_scalar

from scripts.basic import mag, randdirs
from scripts.globals import ma, maGHz


def rc(NS, position, time, exact=False):    # Estimated conversion radius in some direction
    dir = position/mag(position)
    if not exact:
        def to_min(x):
            return NS.wp(x*dir, time) - maGHz
        
    try:
        rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius]).root
        if rc > NS.radius and rc < 100*NS.radius:
            return rc
    except ValueError:
        return None

def conv_surf(NS, time, nsamples=10_000, exact=False):
    points = np.empty((nsamples, 3))
    dirs = randdirs(nsamples)
    
    count = 0
    for ii in range(nsamples):
        rcii = rc(NS, dirs[ii], time, exact=exact)
        if rcii is not None:
            points[ii-count] = rcii*dirs[ii]
        else:
            count += 1

    return points
