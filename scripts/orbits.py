import numpy as np
from numba import njit

from scripts.basic import mag


@njit
def update_ps(p, NS, rprecision=1e-3):
    # Update accelerations
    p.accelerations = NS.grav_field(p.positions)

    # Choosing time steps carefully so that the percent changes in positions and velocities are at most rprecision
    dts = np.empty_like(p.times)
    for ii in range(len(p.times)):
        dts[ii] = rprecision*min([mag(p.velocities)[ii]/mag(p.accelerations)[ii], mag(p.positions)[ii]/mag(p.velocities)[ii]])
        
    # Update velocities and positions
    p.verlet(dts, True)
    # Update accelerations again
    p.accelerations = NS.grav_field(p.positions)
    # Update velocities again
    p.verlet(dts, False)

    p.times += dts
