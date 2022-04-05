import numpy as np
from numba import njit

from scripts.basic import mag
from scripts.globals import G, ma, outdir


# Gravitational energies (10^{-5}eV*(km/s)^2)
@njit
def grav_en(p: object, NS: object) -> np.ndarray:
    return ma*NS.grav_pot(p.positions)

# Total energies (10^{-5}eV*(km/s)^2)
@njit
def energy(p: object, NS: object) -> np.ndarray:
    return p.kin_en() + grav_en(p, NS)

# Distances of minimum approach for the particles
@njit
def min_approach(p: object, NS: object) -> np.ndarray:
    energies, ang_momenta = energy(p, NS), mag(p.ang_momentum())
    rmins = -G*NS.mass*ma/(2*energies) + np.sign(energies)*0.5*np.sqrt((G*NS.mass*ma/energies)**2 + 2*ma*ang_momenta**2/energies)
    return rmins

# Remove particles that are not gonna reach rmax
@njit
def rm_far(p: object, NS: object, rmax: float) -> None:
    indices = np.where(min_approach(p, NS) > rmax)[0]
    p.rm_ps(indices)

# Implementation of Verlet's method to update position and velocity (for reference, see Velocity Verlet in https://en.m.wikipedia.org/wiki/Verlet_integration)
@njit
def update_ps(p: object, NS: object, rprecision: float = 1e-3) -> None:
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

# Full trajectories, use batches of 100 particles for max speed
def trajs(p: object, NS: object, rlimits: np.ndarray = None, fname: str = None, rprecision: float = 1e-3) -> None:
    rmax = max(mag(p.positions))
    
    data = [[] for i in range(8)]
    while max(mag(p.positions)) <= rmax:
        ps_in = np.where(np.logical_and(mag(p.positions) > rlimits[0], mag(p.positions) < rlimits[1]))[0]
        # Save in the format [tags, times, rx, ry, rz, vx, vy, vz]
        data[0].extend(ps_in)
        data[1].extend(p.times[ps_in])
        data[2].extend(p.positions.T[0][ps_in])
        data[3].extend(p.positions.T[1][ps_in])
        data[4].extend(p.positions.T[2][ps_in])
        data[5].extend(p.velocities.T[0][ps_in])
        data[6].extend(p.velocities.T[1][ps_in])
        data[7].extend(p.velocities.T[2][ps_in])
        
        update_ps(p, NS, rprecision=rprecision)

    data = np.array(data).T
    data = data[data[:, 0].argsort()]
    
    np.save(outdir + fname, data)
