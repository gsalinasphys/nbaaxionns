import numpy as np
from numba import float64  # import the types
from numba.experimental import jitclass
from scipy.optimize import root_scalar
from scripts.basic import heav, mag, mydot, nums_vs
from scripts.globals import G, ma, maGHz

spec = [
    ('mass', float64),
    ('radius', float64),
    ('T', float64),
    ('axis', float64[:]),
    ('B0', float64),
    ('misalign', float64),
    ('psi0', float64),
]

# A neutron star
@jitclass(spec)
class NeutronStar:
    def __init__(self, mass=1., radius=10., T=1., axis=np.array([0.,0.,1.]), B0=1., misalign=0., psi0=0.) -> None:
        self.mass = mass            # Neutron star mass (M_Sun)
        self.radius = radius        # Neutron star radius (km)
        self.T = T                  # Period of rotation (seconds)
        self.axis = axis            # Axis of rotation
        self.B0 = B0                # Magnetic field at the surface (10^14 G)
        self.misalign = misalign    # Misalignement angle
        self.psi0 = psi0            # Initial azimuthal angle
    
    # Neutron star's gravitational field (km/s^2) at given positions
    def grav_field(self, positions: np.ndarray) -> np.ndarray:
        ds = mag(positions)
        return -G*self.mass*nums_vs(heav(ds - self.radius, 1.)/ds**3, positions) - G*self.mass/self.radius**3*nums_vs(heav(-ds + self.radius, 0.), positions)

   # Find the gravitational potential produced by the neutron star at some position in (km/s)^2
    def grav_pot(self, positions):
        ds = mag(positions)
        return -G*self.mass*heav(ds - self.radius, 1.)/ds - G*self.mass*((self.radius**2 - ds**2)/(2*self.radius**3) + 1/self.radius)*heav(-ds + self.radius, 0.) 

    # Magnetic dipole with magnitude in units of (10^14 G)*km^3
    def m(self, time):
        psi = self.psi0 + 2*np.pi/self.T*time
        return 0.5*self.B0*self.radius**3*np.array([np.sin(self.misalign)*np.sin(psi), np.sin(self.misalign)*np.cos(psi), np.cos(self.misalign)])

    def B(self, position, time):   # In units of 10^14 Gauss, just the dipole contribution
        d = mag(position)
        return 3*mydot(position, self.m(time))/d**5*position - self.m(time)/d**3

    def wp(self, position, time):  # Plasma frequency in GHz
        return 1.5e2*np.sqrt(np.abs(mydot(self.B(position, time), self.axis))/self.T)
    
    def rcmax(self):
        return 28.231 * self.radius * (self.B0/(self.T*maGHz**2))**(1/3)

