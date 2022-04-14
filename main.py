import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (cylmax, energy, grav_en, mag, metropolis, min_approach,
                     plot_trajs, randdir3d, randdirs3d, rc, rdistr, repeat,
                     rm_far, roche, selectrvs, single_trajs, trajAC, trajs,
                     update_ps)


# Separate __repr__ functions,as Numba does not allow for them
def repr(O: object) -> str:
    if isinstance(O, NeutronStar):
        return (
f"""---------- Neutron Star properties ----------
Mass:                       {O.mass} M_Sun
Radius:                     {O.radius} km
Period:                     {O.T} s
Rotation axis:              {O.axis}
B0:                         {O.B0} x 10^14 G
Misalignement:              {O.misalign} rad
Psi0:                       {O.psi0} rad
Max conversion radius:      {O.rcmax():.4} km
"""
        )
    elif isinstance(O, AxionMiniclusterNFW):
        return (
f"""---------- Axion Clump properties ----------
Clump type:                 Axion Minicluster (NFW profile)
Mass:                       {O.mass} x 10^(-10) M_Sun
Truncation Radius:          {O.rtrunc():.2e} x km
Delta:                      {O.delta}
Concentration:              {O.c}
"""
        )



def runtrajs(nps: int, b: float, vin: np.ndarray, 
             ACparams: tuple = (1, 1., 1.55, 100., 1), lbounds: tuple = (-1., 1.),
             NSparams: tuple = (1.,10.,1.,np.array([1.,0.,1.]),1.,0.,0.),
             rprecision: float = 5e-2) -> np.ndarray:
    if ACparams[0]:
        ACmass, delta, c, vdisptype = ACparams[1:]
        AC = AxionMiniclusterNFW(mass=ACmass, delta=delta, c=c, vdisptype=vdisptype)
    
    NSmass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=NSmass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    
    rvsin = np.array([b*AC.rtrunc(), 0., 1e16]), vin
    pAC = Particles(np.array([rvsin[0]]), np.array([rvsin[1]]))
    trajAC(pAC, NS, roche(AC, NS))
    AC.rCM, AC.vCM = pAC.positions[0], pAC.velocities[0]
    
    rvsinps = selectrvs(AC, NS, nps, lbounds)
    ps = Particles(rvsinps[0], rvsinps[1])
    
    return trajs(ps, NS, rlimits=(NS.radius, NS.rcmax()), rprecision=rprecision)

def main() -> None:
    start = time.perf_counter()
    
    b = 0.2
    vin = np.array([0., 0., -200.])
    nps = 160
    lbounds = (-1./10, 1./10)
    
    ncores = mp.cpu_count() - 1
    with mp.Pool(processes=ncores) as pool:
        result = pool.starmap(runtrajs, [(nps, b, vin, lbounds) for _ in range(2*ncores)])
        pool.close()
        pool.join()
    
    
    end = time.perf_counter()
    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
