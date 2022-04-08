import numpy as np
from numba import float64  # import the types
from numba.experimental import jitclass
from scripts.basic import mag, nums_vs, repeat
from scripts.globals import ma

spec = [
    ('positions', float64[:,:]),
    ('velocities', float64[:,:]),
    ('accelerations', float64[:,:]),
    ('times', float64[:]),
]

@jitclass(spec)
class Particles:
    def __init__(self, positions, velocities, acceleration=np.zeros(3), time=0.) -> None:
        self.positions = positions      # km
        self.velocities = velocities    # km/s
        self.accelerations = repeat(acceleration, len(positions))     # km/s^2
        self.times = np.repeat(time, len(positions))    # s

    # Kinetic energies (10^{-5}eV*(km/s)^2)
    def kin_en(self) -> np.ndarray:
        return 0.5 * ma * np.sum(np.square(self.velocities), axis=1)

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
