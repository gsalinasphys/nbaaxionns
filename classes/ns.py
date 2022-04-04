import numpy as np
from numba import float64  # import the types
from numba.experimental import jitclass
from scripts import G, heav, mag, nums_vs

spec = [
    ('mass', float64),
    ('radius', float64),
    ('period', float64),
    ('axis', float64[:]),
    ('B0', float64),
    ('misalign', float64),
    ('psi0', float64),
]

# A neutron star
@jitclass(spec)
class NeutronStar:
    def __init__(self, mass=1., radius=10., period=1., axis=np.array([0.,0.,1.]), B0=1., misalign=0., psi0=0.) -> None:
        self.mass = mass            # Neutron star mass (solar masses)
        self.radius = radius        # Neutron star radius (km)
        self.period = period        # Period of rotation (seconds)
        self.axis = axis            # Axis of rotation
        self.B0 = B0                # Magnetic field at the surface (10^14 G)
        self.misalign = misalign    # Misalignement angle
        self.psi0 = psi0            # Initial azimuthal angle
    
    # Neutron star's gravitational field (km/s^2) at given positions
    def grav_field(self, positions: np.ndarray) -> np.ndarray:
        ds = mag(positions)
        out = ds > self.radius
        return -G*self.mass*nums_vs(heav(out)/ds**3, positions) - G*self.mass/self.radius**3*nums_vs(heav(np.logical_not(out)), positions)

   # Find the gravitational potential produced by the neutron star at some position in (km/s)^2
    def grav_pot(self, positions):
        ds = mag(positions)
        out = ds > self.radius
        return -G*self.mass*heav(out)/ds - G*self.mass*((self.radius**2 - ds**2)/(2*self.radius**3) + 1/self.radius)*heav(np.logical_not(out)) 
