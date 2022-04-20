from math import pi, sqrt

import numpy as np
from numba import float64, int8
from numba.experimental import jitclass
from scripts.basic import cases, heav, mag, randdir3d, step
from scripts.globals import G, rho_eq

spec = [
    ('rCM', float64[:]),
    ('vCM', float64[:]),
    ('mass', float64),
    ('delta', float64),
    ('c', float64),
    ('vdisptype', int8)
]

@jitclass(spec)
class AxionMiniclusterNFW:
    def __init__(self, rCM=np.zeros(3), vCM=np.zeros(3), mass=1., delta=1.55, c=100., vdisptype=1):
        self.rCM = rCM                      # Position (km) of center of mass
        self.vCM = vCM                      # Velocity (km/s) of center of mass
        self.mass = mass                    # Axion minicluster mass (10^{-10} M_Sun)
        self.delta = delta                  # Parameter delta for NFW profile
        self.c = c                          # Concentration for NFW profile
        self.vdisptype = vdisptype          # Type of velocity dispersion curve: 0 is no dispersion, 1 is Maxwell-Boltzmann

    def rho_s(self) -> float:    # In 10^{-10}*M_Sun/km^3
        return 140*(1 + self.delta)*np.power(self.delta, 3)*rho_eq

    def rs(self) -> float:   # In km
        f_NFW = np.log(1 + self.c) - self.c/(1 + self.c)
        return (self.mass/(4*pi*self.rho_s()*f_NFW)) ** (1/3)

    def rtrunc(self) -> float:  # In km
        return self.c*self.rs()

    # Density profile in units of 10^{-10}*M_Sun/km^3
    def rho_prf(self, positions: np.ndarray, rbounds: tuple = (0., 1.)) -> np.ndarray:
        if isinstance(positions, float):
            d = mag(positions)   # Assumes the clump is at the origin
            return self.rho_s() / (d/self.rs()*(1 + d/self.rs())**2) * step(d/self.rtrunc(), rbounds)
        
        ds = mag(positions - self.rCM)  
        return self.rho_s() / (ds/self.rs()*(1 + ds/self.rs())**2) * step(ds/self.rtrunc(), rbounds)

    def grav_pot(self, positions: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
        if isinstance(positions, float):
            d = mag(positions)   # Assumes the clump is at the origin
            return -4e-10*pi*G*self.rho_s()*self.rs()**3/d*np.log((d + self.rs())/self.rs())
        
        ds = mag(positions - self.rCM)
        return -4e-10*pi*G*self.rho_s()*self.rs()**3/ds*np.log((ds + self.rs())/self.rs())
    
    # Enclosed mass from a given position, in units of 10^{-10} M_Sun
    def encl_mass(self, positions: np.ndarray) -> np.ndarray:
        if isinstance(positions, float):
            d = mag(positions)   # Assumes the clump is at the origin
            return cases(d-self.rtrunc(),
                        4*pi*self.rho_s()*self.rs()**3*(np.log((d + self.rs())/self.rs())-d/(d + self.rs())),
                        self.mass)
        
        ds = mag(positions - self.rCM)
        return cases(ds-self.rtrunc(),
                    4*pi*self.rho_s()*self.rs()**3*(np.log((ds + self.rs())/self.rs())-ds/(ds + self.rs())),
                    self.mass)
        
    # Escape velocity in km/s
    def vesc(self, position: np.ndarray) -> float:
        return sqrt(abs(2*self.grav_pot(position)))

    # Circular velocity in km/s
    def vcirc(self, position: np.ndarray) -> float:
        if isinstance(position, float): # Assumes the clump is at the origin
            return 1e-5*sqrt(G*self.encl_mass(position)/position) * heav(self.rtrunc()-position, 1.)
        
        rstoCM = position - self.rCM
        return 1e-5*sqrt(G*self.encl_mass(position)/mag(rstoCM)) * heav(self.rtrunc()-mag(rstoCM), 1.)

    # Maximum value of velocity dispersion
    def deltav(self, prec: int = 100_000) -> float:
        if not self.vdisptype:
            return 0.
        
        rs = np.linspace(1e-8, 1., prec)*self.rtrunc()
        vscirc = np.empty_like(rs)
        for ii in range(prec):
            vscirc[ii] = self.vcirc(rs[ii])
            
        return max(vscirc)

    # Velocity dispersion at a given position inside the minicluster
    def vdisp(self, position: np.ndarray) -> np.ndarray:
        found = False
        while not found:
            if self.vdisptype == 1: # Maxwell-Boltzmann
                vesc, vcirc = self.vesc(position), self.vcirc(position)
                v_try = np.random.normal(0, vcirc, 3)
                if mag(v_try) < vesc:
                    found = True
        
        return mag(v_try)*randdir3d()
    
    # Velocity dispersion for many positions inside the minicluster
    def vsdisp(self, positions: np.ndarray) -> np.ndarray:
        for ii in range(len(positions)):
            yield self.vdisp(positions[ii])
