import numpy as np


# Roche radius (km), NS mass in M_Sun and axion clump mass in 10^{-10}M_Sun
def roche(AC, NS):
    return AC.rtrunc()*np.power(2*NS.mass/(1e-10*AC.mass), 1/3)
