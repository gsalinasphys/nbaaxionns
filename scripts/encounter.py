import numpy as np
from numba import njit

from scripts.basic import mag
from scripts.orbits import update_ps


# Roche radius (km), NS mass in M_Sun and axion clump mass in 10^{-10}M_Sun
def roche(AC, NS):
    return AC.rtrunc()*np.power(2*NS.mass/(1e-10*AC.mass), 1/3)

# Full trajectories, use batches of 100 particles for max speed, jitted is slower
def trajAC(pAC: object, NS: object, rmin: float = None, rprecision: float = 1e-3) -> None:
    finished = False
  
    while not finished:
        if mag(pAC.positions[0]) < rmin:
            finished = True
        
        update_ps(pAC, NS, rprecision=rprecision)

    return pAC.positions[0], pAC.velocities[0]
