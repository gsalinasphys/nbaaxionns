local_run = True

eV_GHz = 2.41812e5

G = 1.325e11            # Newton's constant (km^3/(M_Sun s^2))
ma = 1           # Axion mass (10^{-5} eV)
maGHz = 1e-5 * eV_GHz * ma
rho_eq = 5.46e-28       # Energy density at MR equality in 10^{-10}*M_Sun/km^3


if local_run:
    outdir = '/home/gsalinas/Dropbox/output/axionns/'
else:
    outdir = '/cfs/data/guvi3498/'
