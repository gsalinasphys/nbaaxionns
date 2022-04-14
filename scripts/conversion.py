from typing import Generator

import numpy as np
from scipy.optimize import root, root_scalar

from scripts.basic import mag, nums_vs, randdirs3d, repeat
from scripts.globals import maGHz

# def rc(time: float, NS: object, position: np.ndarray, exact: bool = False) -> float:    # Estimated conversion radius in some direction
#     dir = position/mag(position)
#     if not exact:
#         def to_min(x: float) -> float:
#             return NS.wp(x*dir, time) - maGHz
        
#     try:
#         rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius], xtol=1e-10, rtol=1e-10).root
#         if rc > NS.radius and rc < 100*NS.radius:
#             return rc
#     except ValueError:
#         return None

def rc(times: float, NS: object, positions: np.ndarray, exact: bool = False) -> float:    # Estimated conversion radius in some direction
    if isinstance(times, float):
        dir = positions/mag(positions)
        if not exact:
            def to_min(x: float) -> float:
                return NS.wp(x*dir, times) - maGHz
            
        try:
            rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius], xtol=1e-10, rtol=1e-10).root
            if rc > NS.radius and rc < 100*NS.radius:
                return rc
        except ValueError:
            return None
    
    dirs = nums_vs(1./mag(positions), positions)
    if not exact:
        def toroot(xs: float) -> float:
            return NS.wp(nums_vs(xs, dirs), times) - maGHz
        
    rc = root(toroot, np.repeat(NS.rcmax(), len(positions))).x
    return rc

def conv_surf(NS: object, time: float, nsamples: int = 10_000, exact: bool = False) -> Generator:    
    for dir in randdirs3d(nsamples):
        rcii = rc(NS, dir, time, exact=exact)
        if rcii is not None:
            yield rcii*dir
