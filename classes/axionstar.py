from math import pi

import numpy as np
from numba import float64, int8
from numba.experimental import jitclass
from scipy.interpolate import interp1d
from scripts.basic import mag
from scripts.globals import ma

spec = [
    ('rCM', float64[:]),
    ('vCM', float64[:]),
    ('mass', float64),
    ('vdisptype', int8)
]

@jitclass(spec)
class AxionStar:
    def __init__(self, rCM=np.zeros(3), vCM=np.zeros(3), mass=1., vdisptype=0):
        self.rCM = rCM                      # Position (km) of center of mass
        self.vCM = vCM                      # Velocity (km/s) of center of mass
        self.mass = mass                    # Axion star mass (10^{-10} M_Sun)
        self.vdisptype = vdisptype          # Type of velocity dispersion curve: 0 is no dispersion
        
    # Radius (km) that contains 99% of the axion star's mass
    def R99(self):
        return 2.64e3/(ma**2*self.mass)
    
    # Truncation radius is chosen at 2R99
    def rtrunc(self):
        return 2*self.R99()
    
    def rho_prf(self, positions: np.ndarray, precision: int = 1_000) -> np.ndarray:
        pdistrib = np.load("input/AS_profile_2R99.npy")
        rs = np.linspace(0., self.rtrunc()/len(pdistrib), precision)
        pdistrib = interp1d(rs, pdistrib)
        norm = self.mass/np.trapz(4*pi*rs**2*pdistrib, rs)
        
        if isinstance(positions, float):    # Assumes the axion star is at the origin
            return pdistrib(positions)/norm
        
        ds = mag(positions - self.rCM)
        return pdistrib(ds)/norm
