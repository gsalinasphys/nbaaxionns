import itertools
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.interpolate import interp1d

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

# Calculated distances of minimum approach for the particles
@njit
def min_approach(p: object, NS: object) -> np.ndarray:
    return -G*NS.mass*ma/(2*energy(p, NS)) + 0.5*np.sqrt((G*NS.mass*ma/energy(p, NS))**2 + 2*ma*mag(p.ang_momentum())**2/energy(p, NS))

# Simulated distances of minimum approach for the particles, compare with above to test quality of orbits
@njit
def min_approach_sim(traj: np.ndarray) -> float:
    return min(mag(traj.T[1:4].T))

# Add particles given their positions, velocities, accelerations and times
@njit
def add_ps(p: object, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, times: np.ndarray, nperiods: np.ndarray) -> None:
    if not p.positions.size:
        p.positions, p.velocities, p.accelerations, p.times, p.nperiods = positions, velocities, accelerations, times, nperiods
    else:
        p.positions = np.concatenate((p.positions, positions))
        p.velocities = np.concatenate((p.velocities, velocities))
        p.accelerations = np.concatenate((p.accelerations, accelerations))
        p.times = np.append(p.times, times)
        p.nperiods = np.append(p.nperiods, nperiods)

# Remove particles that are not gonna reach rcmax
def rm_far(p: object, NS: object) -> None:
    rm_ps(p, np.where(min_approach(p, NS) > NS.rcmax())[0])
    
# Remove particles by their indices
def rm_ps(p: object, inds: int) -> None:
    p.positions = np.delete(p.positions, inds, axis=0)
    p.velocities = np.delete(p.velocities, inds, axis=0)
    p.accelerations = np.delete(p.accelerations, inds, axis=0)
    p.times = np.delete(p.times, inds)
    p.nperiods = np.delete(p.nperiods, inds)

# Implementation of Verlet's method to update position and velocity (for reference, see Velocity Verlet in https://en.m.wikipedia.org/wiki/Verlet_integration)
@njit
def update_ps(p: object, NS: object, rprecision: float = 5e-2) -> None:
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
    
    # Reset clock every period
    period = NS.T
    if np.any(p.times > period):
        for ii in range(len(p.times)):
            p.nperiods[ii] += p.times[ii]//period
        p.times %= period
   
# Full trajectories, use batches of 160 particles for max speed (maybe in general 10ncores?)
def trajs(p: object, NS: object, rlimits: Tuple = None, rprecision: float = 5e-2, fname: str = None) -> None:
    finished = False
    
    data = [[] for i in range(9)]
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
        data[8].extend(p.nperiods[ps_in])
        
        update_ps(p, NS, rprecision=rprecision)

    data = np.array(data).T
    data = data[data[:, 0].argsort()]
    
    if fname:
        np.save(outdir + fname, data)
    
    return data

# Slice into single trajectories
def singletrajs(trajs: np.ndarray) -> np.ndarray:
    key = lambda x: x[0]
    for data in itertools.groupby(trajs, key):
        yield np.array(list(data[1]))[:, 1:]   # Remove slicing to get tags for the particles

# Time order trajectories
@njit
def torder(traj: np.ndarray, tfix: float = None) -> None:
    if tfix:
        tmin, tmax = min(traj.T[0]), max(traj.T[0])
        
        if tmax-tmin > 0.9*tfix:
            for ii in range(len(traj.T[0])):
                if traj.T[0][ii] > tfix/2:
                    traj.T[0][ii] -= tfix

    return traj[traj[:, 0].argsort()]

# Interpolate trajectories
def smoothtraj(traj: np.ndarray, tfix: float = None):
    try:
        traj = torder(traj, tfix)
        return traj.T[0], interp1d(traj.T[0], traj.T[1:4], kind=11), interp1d(traj.T[0], traj.T[4:7], kind=11)
    except (TypeError, ValueError):
        return None

# Plot trajectories
def plot_traj(traj: np.ndarray, show: bool = False, tfix: float = None) -> None:
    ax = plt.axes()
    ax.scatter(traj.T[3], traj.T[1], s=1, linewidths=0)
    
    try:
        tmin, tmax = min(traj.T[0]), max(traj.T[0])
        ts = np.linspace(tmin, tmax, 1000)
        smtraj = smoothtraj(traj, tfix)[1](ts)
        ax.plot(smtraj[2], smtraj[0], lw=.1)
        
    except (TypeError, ValueError):
        pass

    plt.xlabel('$z$ (km)', fontsize=16)
    plt.ylabel('$x$ (km)', fontsize=16)
    
    if show:
            plt.show()
    
# Plot trajectories
def plot_trajs(trajs: np.ndarray, NS: object, fname: str = None, nmax: int = 1_000, show: bool = False) -> None:
    ax = plt.axes()
    ax.set_aspect('equal')

    for traj in trajs[:nmax]:
        ax.scatter(traj.T[3], traj.T[1], s=0.1, linewidths=0)
        
        try:
            tmin, tmax = min(traj.T[0]), max(traj.T[0])
            ts = np.linspace(tmin, tmax, 1000)
            smtraj = smoothtraj(traj, NS.T)[1](ts)
            ax.plot(smtraj[2], smtraj[0], lw=.1)
            
        except (TypeError, ValueError):
            pass
        
    circle = plt.Circle((0, 0), NS.radius, facecolor='purple', alpha = 0.75)
    ax.add_patch(circle)

    plt.xlabel('$z$ (km)', fontsize=16)
    plt.ylabel('$x$ (km)', fontsize=16)
    
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
        
    plt.close()
