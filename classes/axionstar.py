from math import pi

import numpy as np
from numba import float64, int8, int64
from numba.experimental import jitclass
from scipy.interpolate import interp1d
from scripts.basic import heav, mag, nearest
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
    
    def rho_prf(self, positions: np.ndarray, prf: np.ndarray) -> np.ndarray:
        # toret = np.empty(len(positions))
        inds = mag(positions)/self.rtrunc()*len(prf)
        print(inds)
        inds = inds*heav(len(prf)-inds, 1.) - heav(inds-len(prf), 0.)
        print(inds)
        prf = np.append(prf, 0.)
        
        return prf[inds.astype(int64)]
        # for ii in range(len(positions)):
        #     if inds[ii] < len(prf):
        #         return prf[inds.astype(int64)]
        #     else:
        #         return 0.
        
