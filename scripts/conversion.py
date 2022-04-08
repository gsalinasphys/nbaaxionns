from typing import Generator

import numpy as np
from scipy.optimize import root_scalar

from scripts.basic import mag, randdirs3d
from scripts.globals import ma, maGHz


def rc(NS: object, position: np.ndarray, time: float, exact: bool = False) -> float:    # Estimated conversion radius in some direction
    dir = position/mag(position)
    if not exact:
        def to_min(x: float) -> float:
            return NS.wp(x*dir, time) - maGHz
        
    try:
        rc = root_scalar(to_min, bracket=[NS.radius, 100*NS.radius]).root
        if rc > NS.radius and rc < 100*NS.radius:
            return rc
    except ValueError:
        return None

def conv_surf(NS: object, time: float, nsamples: int = 10_000, exact: bool = False) -> Generator:    
    for dir in randdirs3d(nsamples):
        rcii = rc(NS, dir, time, exact=exact)
        if rcii is not None:
            yield rcii*dir
