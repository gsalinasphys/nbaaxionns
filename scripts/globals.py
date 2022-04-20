from numba.experimental import jitclass

local_run = True

G = 1.325e11                    # Newton's constant (km^3/(M_Sun s^2))
c = 2.99792458*1e5              # Speed of light in km/s
ma = 1                          # Axion mass (10^{-5} eV)
rho_eq = 5.46e-28               # Energy density at MR equality in 10^{-10}*M_Sun/km^3
yr = 3.154e7                    # Year in seconds
Msun = 1.1157e+66               # Solar mass in eV
gag = 1.                        # Axion-photon coupling (10^{-14} GeV^{-1})

# Conversion factors
eV_GHz = 2.41812e5
km_eVinv = 5.067728930e9
G_eV2 = 0.0692507779
MSun_eV = 1.115746686e66

maGHz = 1e-5 * eV_GHz * ma      # Axion mass in GHz

if local_run:
    outdir = 'output/'
else:
    outdir = '/cfs/data/guvi3498/nbaaxionns/'

@jitclass
class EmptyClass:
    def __init__(self):
        pass
