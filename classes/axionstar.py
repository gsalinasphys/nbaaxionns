from math import pi, sqrt

import numpy as np
from numba import float64, int8, int64
from numba.experimental import jitclass
from scipy.interpolate import interp1d
from scripts.basic import heav, mag, myint
from scripts.globals import G, ma

spec = [
    ('rCM', float64[:]),
    ('vCM', float64[:]),
    ('mass', float64),
    ('vdisptype', int8)
]

@jitclass(spec)
class AxionStar:
    def __init__(self, rCM=np.zeros(3), vCM=np.zeros(3), mass=.01, vdisptype=0):
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
    
    # Density profile in units of 10^{-10}*M_Sun/km^3
    def rho_prf(self, positions: np.ndarray, prf: np.ndarray) -> np.ndarray:
        rs = np.linspace(0, self.rtrunc(), len(prf))
        norm = self.mass/myint(4*pi*rs**2*prf, rs)
        if isinstance(positions, float):
            return norm*prf[int64(mag(positions)/self.rtrunc()*len(prf))]
            
        inds = mag(positions-self.rCM)/self.rtrunc()*len(prf)
        inds = inds*heav(len(prf)-inds, 1.) - heav(inds-len(prf), 0.)
        prf = np.append(prf, 0.)
        
        return norm*prf[inds.astype(int64)]
    
    # Enclosed mass given position
    def encl_mass(self, positions: np.ndarray, prf: np.ndarray) -> np.ndarray:
        rs = np.linspace(0, self.rtrunc(), len(prf))
        norm = self.mass/myint(4*pi*rs**2*prf, rs)
        if isinstance(positions, float):
            ind = int64(mag(positions)/self.rtrunc()*len(prf))
            return norm*myint(4*pi*rs[:ind]**2*prf[:ind], rs[:ind])
        inds = mag(positions-self.rCM)/self.rtrunc()*len(prf)
        inds = inds*heav(len(prf)-inds, 1.) - heav(inds-len(prf), 0.)

        toret = np.empty(len(positions))
        for ii in range(len(positions)):
            ind = int64(inds[ii])
            if ind != -1:
                toret[ii] = norm*myint(4*pi*rs[:ind]**2*prf[:ind], rs[:ind])
            else:
                toret[ii] = self.mass
        
        return toret
    
    def grav_pot(self, positions: np.ndarray, prf: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
        rs = np.linspace(0, self.rtrunc(), len(prf))
        norm = self.mass/myint(4*pi*rs**2*prf, rs)
        if isinstance(positions, float):
            ind = int64(mag(positions)/self.rtrunc()*len(prf))
            return -G*self.encl_mass(positions, prf)/positions - G*norm*myint(4*pi*rs[ind:]*prf[ind:], rs[ind:])
        ds = mag(positions-self.rCM)
        inds = ds/self.rtrunc()*len(prf)
        inds = inds*heav(len(prf)-inds, 1.) - heav(inds-len(prf), 0.)
    
        encl_ms = self.encl_mass(positions, prf)
        toret = np.empty(len(positions))
        for ii in range(len(positions)):
            ind = int64(inds[ii])
            if ind != -1:
                toret[ii] = -G*encl_ms[ii]/ds[ii] - G*norm*myint(4*pi*rs[ind:]*prf[ind:], rs[ind:])
            else:
                toret[ii] = -G*encl_ms[ii]/ds[ii]
        
        return toret

    # Escape velocity in km/s
    def vesc(self, position: np.ndarray) -> float:
        return sqrt(abs(2*self.grav_pot(position)))
        