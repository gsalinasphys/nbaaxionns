from numba.experimental import jitclass

local_run = True

eV_GHz = 2.41812e5

G = 1.325e11                    # Newton's constant (km^3/(M_Sun s^2))
ma = 1                          # Axion mass (10^{-5} eV)
maGHz = 1e-5 * eV_GHz * ma      # Axion mass in GHz
rho_eq = 5.46e-28               # Energy density at MR equality in 10^{-10}*M_Sun/km^3
yr = 3.154e7                    # Year in seconds
Msun = 1.1157e+66               # Solar mass in eV
gag = 1.                        # Axion-photon coupling (10^{-14} GeV^{-1})

if local_run:
    outdir = 'output/'
else:
    outdir = '/cfs/data/guvi3498/'

@jitclass
class EmptyClass:
    def __init__(self):
        pass
