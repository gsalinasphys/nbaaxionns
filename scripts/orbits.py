import itertools
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

from scripts.basic import mag
from scripts.globals import G, ma, outdir, yr


# Gravitational energies (10^{-5}eV*(km/s)^2)
@njit
def grav_en(p: object, NS: object) -> np.ndarray:
    return ma*NS.grav_pot(p.positions)

# Total energies (10^{-5}eV*(km/s)^2)
@njit
def energy(p: object, NS: object) -> np.ndarray:
    return p.kin_en() + grav_en(p, NS)

# Calculated distances of minimum approach for the particles
@njit
def min_approach(p: object, NS: object) -> np.ndarray:
    # energies, ang_momenta = energy(p, NS), mag(p.ang_momentum())
    # rmins = -G*NS.mass*ma/(2*energies) + 0.5*np.sqrt((G*NS.mass*ma/energies)**2 + 2*ma*ang_momenta**2/energies)
    # return rmins
    return -G*NS.mass*ma/(2*energy(p, NS)) + 0.5*np.sqrt((G*NS.mass*ma/energy(p, NS))**2 + 2*ma*mag(p.ang_momentum())**2/energy(p, NS))
    

# Simulated distances of minimum approach for the particles, compare with above to test quality of orbits
@njit
def min_approach_sim(traj: np.ndarray) -> float:
    return min(mag(traj.T[1:4].T))

# Add particles given their positions, velocities, accelerations and times
def add_ps(p: object, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, times: np.ndarray) -> None:
    if not p.positions.size:
        p.positions, p.velocities, p.accelerations, p.times = positions, velocities, accelerations, times
    else:
        p.positions = np.append(p.positions, positions, axis=0)
        p.velocities = np.append(p.velocities, velocities, axis=0)
        p.accelerations = np.append(p.accelerations, accelerations, axis=0)
        p.times = np.append(p.times, times)

# Remove particles by their indices
def rm_ps(p: object, inds: int) -> None:
    p.positions = np.delete(p.positions, inds, axis=0)
    p.velocities = np.delete(p.velocities, inds, axis=0)
    p.accelerations = np.delete(p.accelerations, inds, axis=0)
    p.times = np.delete(p.times, inds)

# Remove particles that are not gonna reach rmax
def rm_far(p: object, NS: object, rmax: float) -> None:
    rm_ps(p, np.where(min_approach(p, NS) > rmax))

# Implementation of Verlet's method to update position and velocity (for reference, see Velocity Verlet in https://en.m.wikipedia.org/wiki/Verlet_integration)
@njit
def update_ps(p: object, NS: object, rprecision: float = 1e-2) -> None:
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
    
    # Reset clock every ten years
    period = 10.*yr
    if np.any(p.times > period):
        p.times -= period

    p.times += dts
    
# Full trajectories, use batches of 160 particles for max speed (maybe in general 10ncores?)
def trajs(p: object, NS: object, rlimits: Tuple = None, fname: str = None, retval: bool = False, rprecision: float = 1e-2) -> None:
    finished = False
    
    data = [[] for i in range(8)]
    while not finished or min(mag(p.positions)) < rlimits[1]:
        if min(mag(p.positions)) < rlimits[1]:
            finished = True
            
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
    
    if fname is not None:
        np.save(outdir + fname, data)
        
    if retval:
        return data

# Slice into single trajectories
def single_trajs(trajs: np.ndarray) -> np.ndarray:
    key = lambda x: x[0]
    for data in itertools.groupby(trajs, key):
        yield np.array(list(data[1]))[:, 1:]   # Remove slicing to get tags for the particles

# Time order trajectories
def torder(traj: np.ndarray) -> None:
    return traj[traj[:, 0].argsort()]

# Plot trajectories
def plot_traj(traj: np.ndarray, show: bool = False) -> None:
    ax = plt.axes()

    ax.scatter(traj.T[2], traj.T[1], s=1)

    plt.xlabel('$y$ (km)', fontsize=16)
    plt.ylabel('$x$ (km)', fontsize=16)

    if show:
        plt.show()
    
# Plot trajectories
def plot_trajs(trajs: np.ndarray, NS: object, fname: str = None, show: bool = False) -> None:
    ax = plt.axes()
    ax.set_aspect('equal')

    for traj in trajs:
        ax.scatter(traj.T[2], traj.T[1], s=1e-2)
        
    circle = plt.Circle((0, 0), NS.radius, facecolor='purple', alpha = 0.5)
    ax.add_patch(circle)

    plt.xlabel('$y$ (km)', fontsize=16)
    plt.ylabel('$x$ (km)', fontsize=16)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
