import time

import numpy as np
from numba import njit

from classes import NeutronStar, Particles
from scripts import mag, nums_vs


@njit
def verlet_step(p, NS, rprecision=1e-3):
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

def main():
    start = time.perf_counter()

    r_in, v_in = [6e3, 1e14, 0], [0, -200., 0]
    p = Particles(np.array([r_in]*10), np.array([v_in]*10))

    NS = NeutronStar()

    print(NS.grav_potential(p.positions))
    # [verlet_step(p, NS) for _ in range(60000)]

    end = time.perf_counter()

    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
