import numpy as np
from numba import float64, int8  # import the types
from numba.experimental import jitclass
from scripts.basic import heav, mag, randdir
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
        return (self.mass/(4*np.pi*self.rho_s()*f_NFW)) ** (1/3)

    def rtrunc(self) -> float:
        return self.c*self.rs()

    def rho_prf(self, positions: np.ndarray, rbounds: tuple = (0., 1.)) -> np.ndarray:   # In units of 10^{-10}*M_Sun/km^3
        if isinstance(positions, float):
            d = positions   # Assumes the clump is at the origin
            return self.rho_s() / (d/self.rs()*(1 + d/self.rs())**2) * (heav(rbounds[1]*self.rtrunc() - d, 1.) - heav(rbounds[0]*self.rtrunc() - d, 1.))
        elif positions.ndim == 1:
            d = mag(positions - self.rCM)
            return self.rho_s() / (d/self.rs()*(1 + d/self.rs())**2) * (heav(rbounds[1]*self.rtrunc() - d, 1.) - heav(rbounds[0]*self.rtrunc() - d, 1.))
        ds = mag(positions - self.rCM)
         
        return self.rho_s() / (ds/self.rs()*(1 + ds/self.rs())**2) * (heav(rbounds[1]*self.rtrunc() - ds, 1.) - heav(rbounds[0]*self.rtrunc() - ds, 1.))

    def grav_pot(self, positions: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
        if isinstance(positions, float):
            d = positions   # Assumes the clump is at the origin
            return -4e-10*np.pi*G*self.rho_s()*self.rs()**3/d*np.log((d + self.rs())/self.rs())
        elif positions.ndim == 1:
            d = mag(positions - self.rCM)
            return -4e-10*np.pi*G*self.rho_s()*self.rs()**3/d*np.log((d + self.rs())/self.rs())
        ds = mag(positions - self.rCM)
        
        return -4e-10*np.pi*G*self.rho_s()*self.rs()**3/ds*np.log((ds + self.rs())/self.rs())
    
    # Enclosed mass from a given position, in units of 10^{-10} M_Sun
    def encl_mass(self, positions: np.ndarray) -> np.ndarray:
        if isinstance(positions, float):
            d = positions   # Assumes the clump is at the origin
            return 4*np.pi*self.rho_s()*self.rs()**3*(np.log((d + self.rs())/self.rs()) - d/(d + self.rs()))*heav(self.rtrunc() - d, 0.) + self.mass*heav(-self.rtrunc() + d, 1.)
        elif positions.ndim == 1:
            d = mag(positions - self.rCM)
            return 4*np.pi*self.rho_s()*self.rs()**3*(np.log((d + self.rs())/self.rs()) - d/(d + self.rs()))*heav(self.rtrunc() - d, 0.) + self.mass*heav(-self.rtrunc() + d, 1.)
        ds = mag(positions - self.rCM)
        
        return 4*np.pi*self.rho_s()*self.rs()**3*(np.log((ds + self.rs())/self.rs()) - ds/(ds + self.rs()))*heav(self.rtrunc() - ds, 0.) + self.mass*heav(-self.rtrunc() + ds, 1.)

    # Escape velocity in km/s
    def vesc(self, position: np.ndarray) -> float:
        return np.sqrt(np.abs(2*self.grav_pot(position)))

    # Circular velocity in km/s
    def vcirc(self, position: np.ndarray) -> float:
        rstoCM = position - self.rCM
        return 1e-5*np.sqrt(G*self.encl_mass(position)/mag(rstoCM))

    # Velocity dispersion at a given position inside the minicluster
    def vdisp(self, position: np.ndarray) -> np.ndarray:
        found = False
        while not found:
            if self.vdisptype == 1: # Maxwell-Boltzmann
                vesc, vcirc = self.vesc(position), self.vcirc(position)
                v_try = np.random.normal(0, vcirc, 3)
                if mag(v_try) < vesc:
                    found = True
            
        return v_try*randdir()
    
    # Velocity dispersion for many positions inside the minicluster
    def vsdisp(self, positions: np.ndarray) -> np.ndarray:
        for ii in range(len(positions)):
            yield self.vdisp(positions[ii])
