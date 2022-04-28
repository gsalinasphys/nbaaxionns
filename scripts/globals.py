from math import pi, sqrt

from numba.experimental import jitclass

local_run = True

ma = 1                          # Axion mass (10^{-5} eV)

G = 1.32712e11                  # Newton's constant (km^3/(M_Sun s^2))
c = 2.99792*1e5                 # Speed of light in km/s
me = 5.0e5                      # Electron mass (eV)
alpha = 1/137                   # Fine structure constant
e = sqrt(4*pi*alpha)            # Electron charge in natural units
hbar = 6.582119e-16             # Planck constant (eV*s)
rho_eq = 5.46e-28               # Energy density at MR equality in 10^{-10}*M_Sun/km^3
Msun = 1.1157e+66               # Solar mass in eV

G_eV2 = 1.95e-2                 # Convert Gauss to eV^2

if local_run:
    outdir = 'output/'
else:
    outdir = '/cfs/data/guvi3498/nbaaxionns/'

@jitclass
class EmptyClass:
    def __init__(self):
        pass
