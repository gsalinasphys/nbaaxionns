import numpy as np
from numba import float64  # import the types
from numba.experimental import jitclass
from scripts import axionmass, mag, nums_vs, rm_inds

spec = [
    ('positions', float64[:,:]),
    ('velocities', float64[:,:]),
    ('accelerations', float64[:,:]),
    ('times', float64[:]),
]

@jitclass(spec)
class Particles:
    def __init__(self, positions, velocities, acceleration=np.array([0., 0., 0.]), time=0.) -> None:
        self.positions = positions      # km
        self.velocities = velocities    # km/s
        self.accelerations = np.repeat(acceleration, len(positions)).reshape(len(positions), 3)     # km/s^2
        self.times = np.repeat(time, len(positions))    # s

    # Kinetic energies (10^{-5}eV*(km/s)^2)
    def kin_en(self) -> np.ndarray:
        return 0.5 * axionmass * np.sum(np.square(self.velocities), axis=1)

    # Angular momenta (km^2/s)
    def ang_momentum(self) -> np.ndarray:
        return np.cross(self.positions, self.velocities)

    # Impact parameter (km)
    def impact_param(self) -> np.ndarray:
        return mag(self.ang_momentum()) / mag(self.velocities)

    # Intermediary updates of positions and velocities for Verlet integration, note that the first update has an additional step
    def verlet(self, dts: np.ndarray, first_update: bool) -> None:
        self.velocities += 0.5*nums_vs(dts, self.accelerations)

        if first_update:
            self.positions += nums_vs(dts, self.velocities)

    # Add particles given their positions, velocities, accelerations and times
    def add_ps(self, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray = np.array([0., 0., 0.]), times: np.ndarray = 0.) -> None:
        if len(self.positions):
            self.positions = np.append(self.positions, positions, axis=0)
            self.velocities = np.append(self.velocities, velocities, axis=0)
            self.accelerations = np.append(self.accelerations, accelerations, axis=0)
            self.times = np.append(self.times, times)
        else:
            self.positions, self.velocities, self.accelerations = positions, velocities, accelerations      
        
    # Remove particles given their indices
    def rm_ps(self, inds: np.ndarray) -> None:
        self.positions = rm_inds(self.positions, inds)
        self.velocities = rm_inds(self.velocities, inds)
        self.accelerations = rm_inds(self.accelerations, inds)
        self.times = rm_inds(self.times, inds)        
