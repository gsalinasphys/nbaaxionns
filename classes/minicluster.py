import numpy as np
from numba import float64, int8  # import the types
from numba.experimental import jitclass
from scripts.basic import heav, mag
from scripts.globals import G, rho_eq

spec = [
    ('rCM', float64[:]),
    ('vCM', float64[:]),
    ('mass', float64),
    ('delta', float64),
    ('c', float64),
]

@jitclass(spec)
class AxionMiniclusterNFW:
    def __init__(self, rCM, vCM, mass=1., delta=1.55, c=100.):
        self.rCM = rCM                      # Position (km) of center of mass
        self.vCM = vCM                      # Velocity (km/s) of center of mass
        self.mass = mass                    # Axion minicluster mass (10^{-10} M_Sun)
        self.delta = delta                  # Parameter delta for NFW profile
        self.c = c                          # Concentration for NFW profile

    def rho_s(self) -> float:    # In units of 10^{-10}*M_Sun/km^3
        return 140*(1 + self.delta)*self.delta**3*rho_eq

    def rs(self) -> float:   # In km
        f_NFW = np.log(1 + self.c) - self.c/(1 + self.c)
        return (self.mass/(4*np.pi*self.rho_s()*f_NFW))**(1/3)

    def rtrunc(self) -> float:
        return self.c*self.rs()

    def rho_prf(self, positions: np.ndarray) -> np.ndarray:   # In units of 10^{-10}*M_Sun/km^3
        rstoCM = positions - self.rCM
        ds = mag(rstoCM)
        return self.rho_s()/(ds/self.rs()*(1 + ds/self.rs())**2)*heav(self.rtrunc() - ds)

    def grav_pot(self, positions: np.ndarray) -> np.ndarray:   # In units of (km/s)^2
        rstoCM = positions - self.rCM
        ds = mag(rstoCM)
        return -4e-10*np.pi*G*self.rho_s()*self.rs()**3/ds*np.log((ds + self.rs())/self.rs())

    # Enclosed mass from a given position, in units of 10^{-10} solar masses
    def encl_mass(self, positions: np.ndarray) -> np.ndarray:
        rstoCM = positions - self.rCM
        return 4*np.pi*self.rho_s()*self.rs()**3*(np.log((mag(rstoCM) + self.rs())/self.rs()) - mag(rstoCM)/(mag(rstoCM) + self.rs()))

    # Escape velocity in km/s
    def v_esc(self, position: np.ndarray) -> float:
        return np.sqrt(np.abs(2*self.grav_pot(position)))

    # Circular velocity in km/s
    def v_circ(self, position: np.ndarray) -> float:
        rstoCM = position - self.rCM
        return 1e-5*np.sqrt(G*self.encl_mass(position)/mag(rstoCM))

    # Velocity dispersion at a given position inside the minicluster
    def vdisp(self, position: np.ndarray) -> np.ndarray:
        found = False
        while not found:
            v_esc, v_circ = self.v_esc(position), self.v_circ(position)
            v_try = np.random.normal(0, v_circ, 3)
            if mag(v_try) < v_esc:
                found = True
            
        return v_try
    
    # Velocity dispersion for many positions inside the minicluster
    def vsdisp(self, positions: np.ndarray) -> np.ndarray:
        drawn_vs = np.empty((len(positions), 3))
        for ii in range(len(positions)):
            drawn_v = self.vdisp(positions[ii])
            for jj in range(3):
                drawn_vs[ii, jj] = drawn_v[jj]
                
        return drawn_vs
