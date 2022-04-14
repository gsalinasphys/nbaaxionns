import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (allhits, cylmax, energy, grav_en, id_gen, mag, metropolis,
                     min_approach, outdir, plot_trajs, randdir3d, randdirs3d,
                     rc, rdistr, repeat, rm_far, roche, selectrvs, singletrajs,
                     trajAC, trajs, update_ps)


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



def run(nps: int, b: float, vin: np.ndarray, 
             ACparams: tuple = (1, 1., 1.55, 100., 1), lbounds: tuple = (-1., 1.),
             NSparams: tuple = (1.,10.,1.,np.array([1.,0.,1.]),1.,0.,0.),
             rprecision: float = 5e-2, padding: float = 10.,
             fnametrajs: str = None, fnamehits: str = None) -> np.ndarray:
    if ACparams[0]:
        ACmass, delta, c, vdisptype = ACparams[1:]
        AC = AxionMiniclusterNFW(mass=ACmass, delta=delta, c=c, vdisptype=vdisptype)
    
    NSmass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=NSmass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    
    rvsin = np.array([b*AC.rtrunc(), 0., 1e16]), vin
    pAC = Particles(np.array([rvsin[0]]), np.array([rvsin[1]]))
    trajAC(pAC, NS, roche(AC, NS), rprecision)
    AC.rCM, AC.vCM = pAC.positions[0], pAC.velocities[0]
    
    rvsinps = selectrvs(AC, NS, nps, lbounds)
    ps = Particles(rvsinps[0], rvsinps[1])
    
    simtrajs = trajs(ps, NS, (NS.radius, NS.rcmax(padding)), rprecision, fnametrajs)
    simtrajs = [np.array(simtraj) for simtraj in list(singletrajs(simtrajs))]
    
    ahits = list(allhits(NS, simtrajs))
    ahits = np.array([hit for subhits in ahits for hit in subhits])
    
    if fnamehits is not None:
        np.save(outdir + fnamehits, ahits)
        
    return ahits

def main() -> None:
    start = time.perf_counter()
    
    nps = 160
    b = 0.2
    vin = np.array([0., 0., -200.])
    ACparams = (1, 1., 1.55, 100., 1)
    lbounds = (-1., 1.)
    NSparams = (1.,10.,1.,np.array([1.,0.,1.]),1.,0.,0.)
    rprecision = 5e-2
    padding = 10.
    ACshort = 'MC' if ACparams[0] else 'AS'
    id = id_gen()
    fnametrajs = ACshort + id + '_trajs'
    fnamehits = ACshort + id + '_conversion'
    
    ncores = mp.cpu_count() - 1
    with mp.Pool(processes=ncores) as pool:
        result = pool.starmap(run, [(nps, b, vin, ACparams, lbounds,
                                     NSparams, rprecision, padding,
                                     fnametrajs+f'{ii}' if fnametrajs is not None else None, 
                                     fnamehits+f'{ii}' if fnamehits is not None else None)
                                    for ii in range(200*ncores)])
        pool.close()
        pool.join()
    
    result = np.concatenate(result)
    print(result)
    
    end = time.perf_counter()
    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
