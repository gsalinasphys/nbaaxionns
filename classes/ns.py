from math import cos, pi, sin, sqrt

import numpy as np
from numba import float64  # import the types
from numba.experimental import jitclass
from scipy.optimize import root_scalar
from scripts.basic import cases, heav, mag, mydot, nums_vs, repeat
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
        if isinstance(positions, float):
            d = positions   # Assumes the neutron star is at the origin
            return cases(d-self.radius,
                        -G*self.mass*d/self.radius**3,
                        -G*self.mass/d**2)
        elif positions.ndim == 1:
            d = mag(positions)
            return cases(d-self.radius,
                        -G*self.mass*positions/self.radius**3,
                        -G*self.mass/d**2*positions)

        ds = mag(positions)
        return -G*self.mass*nums_vs(heav(ds - self.radius, 1.)/ds**3, positions) - G*self.mass/self.radius**3*nums_vs(heav(-ds + self.radius, 0.), positions)

   # Find the gravitational potential produced by the neutron star at some position in (km/s)^2
    def grav_pot(self, positions: np.ndarray) -> np.ndarray:
        if isinstance(positions, float):
            d = positions   # Assumes the neutron star is at the origin
            return cases(d-self.radius, 
                        -G*self.mass*((self.radius**2 - d**2)/(2*self.radius**3) + 1/self.radius),
                        -G*self.mass/d)
        elif positions.ndim == 1:
            d = mag(positions)
            return cases(d-self.radius, 
                        -G*self.mass*((self.radius**2 - d**2)/(2*self.radius**3) + 1/self.radius),
                        -G*self.mass/d)
        
        ds = mag(positions)
        return cases(ds-self.radius, 
                    -G*self.mass*((self.radius**2 - ds**2)/(2*self.radius**3) + 1/self.radius),
                    -G*self.mass/ds)

    # # Magnetic dipole with magnitude in units of (10^14 G)*km^3
    # def m(self, time: float) -> np.ndarray:
    #     psi = self.psi0 + 2*pi/self.T*time
    #     return 0.5*self.B0*self.radius**3*np.array([np.sin(self.misalign)*np.sin(psi), np.sin(self.misalign)*np.cos(psi), np.cos(self.misalign)])

    # def B(self, position: np.ndarray, time: float) -> np.ndarray:   # In units of 10^14 Gauss, just the dipole contribution
    #     d = mag(position)
    #     return 3*mydot(position, self.m(time))/d**5*position - self.m(time)/d**3

    # def wp(self, position: np.ndarray, time: float) -> float:  # Plasma frequency in GHz
    #     return 1.5e2*sqrt(abs(mydot(self.B(position, time), self.axis))/self.T)
    
    # Magnetic dipole with magnitude in units of (10^14 G)*km^3
    def m(self, times: float) -> np.ndarray:
        if isinstance(times, float):
            psi = self.psi0 + 2*pi/self.T*times
            return 0.5*self.B0*self.radius**3*np.array([np.sin(self.misalign)*np.sin(psi), np.sin(self.misalign)*np.cos(psi), np.cos(self.misalign)])

        psis = self.psi0 + 2*pi/self.T*times
        return 0.5*self.B0*self.radius**3*np.concatenate((sin(self.misalign)*np.sin(psis), sin(self.misalign)*np.cos(psis), cos(self.misalign)*np.ones(len(times)))).reshape((3, len(times))).T

    def B(self, positions: np.ndarray, times: float) -> np.ndarray:   # In units of 10^14 Gauss, just the dipole contribution
        if isinstance(times, float):
            d = mag(positions)
            return 3*mydot(positions, self.m(times))/d**5*positions - self.m(times)/d**3
        
        ds = mag(positions)
        return nums_vs(3*mydot(positions, self.m(times))/ds**5, positions) - nums_vs(1./ds**3, self.m(times))


    def wp(self, positions: np.ndarray, times: float) -> float:  # Plasma frequency in GHz
        if isinstance(times, float):
            return 1.5e2*sqrt(abs(mydot(self.B(positions, times), self.axis))/self.T)
        
        return 1.5e2*np.sqrt(np.abs(mydot(self.B(positions, times), repeat(self.axis, len(positions)))/self.T))
    
    # Maximum radius of conversion with some padding (in percent)
    def rcmax(self, padding: float = 0.) -> float:
        return (1+padding/100) * 28.231 * self.radius * (self.B0/(self.T*maGHz**2))**(1/3)

