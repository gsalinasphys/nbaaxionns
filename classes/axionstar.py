import random
from math import pi, sqrt

import numpy as np
from numba import float64, int8, int64
from numba.experimental import jitclass
from scripts.basic import (cases, mag, myint, myintfrom, myintupto, nums_vs,
                           randdirs3d)
from scripts.globals import G, ma

spec = [
    ('rCM', float64[:]),
    ('vCM', float64[:]),
    ('mass', float64),
    ('vdisptype', int8),
    ('prf', float64[:])
]

@jitclass(spec)
class AxionStar:
    def __init__(self, rCM=np.zeros(3), vCM=np.zeros(3), mass=.01, vdisptype=0, prf=np.empty(0)):
        self.rCM = rCM                      # Position (km) of center of mass
        self.vCM = vCM                      # Velocity (km/s) of center of mass
        self.mass = mass                    # Axion star mass (10^{-10} M_Sun)
        self.vdisptype = vdisptype          # Type of velocity dispersion curve: 0 is no dispersion
        self.prf = prf                      # Not-to-scale density profile as input to be rescaled
        
    # Radius (km) that contains 99% of the axion star's mass
    def R99(self):
        return 26.4/(ma**2*self.mass)
    
    # Truncation radius is chosen at 2R99
    def rtrunc(self):
        return 2*self.R99()
    
    # Density profile in units of 10^{-10}*M_Sun/km^3
    def rho_prf(self, positions: np.ndarray, rbound: float = 1.) -> np.ndarray:
        rs = np.linspace(0, self.rtrunc(), len(self.prf))
        norm = self.mass/myint(4*pi*rs**2*self.prf, rs)
        if isinstance(positions, float):    # Assumes axion star is at the origin
            ind = int64(mag(positions)/self.rtrunc()*len(self.prf))
            return norm*self.prf[ind] if ind<len(self.prf) else 0.
    
        return cases(mag(positions-self.rCM)-rbound*self.rtrunc(),
              norm*self.prf[(mag(positions-self.rCM)/self.rtrunc()*len(self.prf)).astype(int64)],
              np.zeros(len(positions)))
    
    # Enclosed mass given position in units of 10^{-10}*M_Sun
    def encl_mass(self, positions: np.ndarray) -> np.ndarray:
        rs = np.linspace(0, self.rtrunc(), len(self.prf))
        norm = self.mass/myint(4*pi*rs**2*self.prf, rs)
        if isinstance(positions, float):    # Assumes axion star is at the origin
            ind = int64(mag(positions)/self.rtrunc()*len(self.prf))
            return norm*myint(4*pi*rs[:ind]**2*self.prf[:ind], rs[:ind])
      
        inds = (mag(positions-self.rCM)/self.rtrunc()*len(self.prf)).astype(int64)
        return cases(mag(positions-self.rCM)-self.rtrunc(),
                     norm*np.array(list(myintupto(4*pi*rs**2*self.prf, rs, inds))),
                     np.repeat(self.mass, len(positions)))
    
    def grav_pot(self, positions: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
        rs = np.linspace(0, self.rtrunc(), len(self.prf))
        norm = self.mass/myint(4*pi*rs**2*self.prf, rs)
        if isinstance(positions, float):    # Assumes axion star is at the origin
            ind = int64(mag(positions)/self.rtrunc()*len(self.prf))
            return -1e-10*G*self.encl_mass(positions)/positions - 1e-10*G*norm*myint(4*pi*rs[ind:]*self.prf[ind:], rs[ind:])
        
        ds = mag(positions-self.rCM)
        inds = (ds/self.rtrunc()*len(self.prf)).astype(int64)
        encl_ms = self.encl_mass(positions)
        return cases(mag(positions-self.rCM)-self.rtrunc(),
                    -1e-10*G*encl_ms/ds - 1e-10*G*norm*np.array(list(myintfrom(4*pi*rs*self.prf, rs, inds))),
                    -1e-10*G*self.mass/ds)
        
    # Escape velocity in km/s
    def vesc(self, positions: np.ndarray) -> float:
        return np.sqrt(np.abs(2*self.grav_pot(positions)))
    
    # Maximum value of velocity dispersion
    def deltav(self) -> float:
        return self.vesc(1.e-4)
    
    # Velocity dispersion for many positions inside the axion star
    def vsdisp(self, positions: np.ndarray) -> np.ndarray:
        if self.vdisptype == 1: # Flat distribution
            return nums_vs(np.random.random(size=len(positions))*self.vesc(positions), randdirs3d(len(positions)))
        