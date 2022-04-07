import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (energy, grav_en, mag, metropolis, min_approach,
                     plot_trajs, randdir, randdirs, rc, rdistr, repeat, rm_far,
                     rm_inds, runtrajs, single_trajs, trajAC, trajs, truntrajs,
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

def main() -> None:
    
    # rin, vin = [6e3, 1e14, 0], [0, -200., 0]
    # nps = 160
    # MC = AxionMiniclusterNFW(np.array(rin), np.array(vin))
    # pMC = Particles(MC.rCM.reshape((1,3)), MC.vCM.reshape((1,3)))
    # NS = NeutronStar()
    
    # print(repr(NS))
    # print(repr(MC))
        
    # samples = mag(np.array(list(metropolis(MC, rdistr, 1_000_000))))
    # plt.hist(np.abs(samples), bins=np.linspace(0,1.,1000))
    # plt.show()
    # plt.close()
    
    start = time.perf_counter()
    
    nps = (16*16 + 1)
    rsin = np.array([[np.random.random()*1e2, 1e14, 0] for _ in range(1, nps+1)])
    vsin = repeat(np.array([0, -200., 0.]), nps)
    rvsin = (rsin, vsin)
    rmax = 100.
    NSparams = (1.,10.,1.,np.array([1.,0.,1.]),1.,0.,0.)
    rprecision = 1e-2
    
    ncores = mp.cpu_count() - 1
    with mp.Pool(processes=ncores) as pool:
        result = pool.starmap(truntrajs, [(runtrajs, rvsin, rmax, NSparams, rprecision) for _ in range(ncores)])
        pool.close()
        pool.join()
        
    print(result[0])
    print(result[0][-1, 0])
    print(result[0].shape)
    
    # print(trajAC(pMC, NS, 100.))
    # print(pMC.positions[0], pMC.velocities[0])
    
    # p = Particles(np.array([rin]*nps), np.array([vin]*nps))
    # p2 = Particles(np.array([rin]*nps), np.array([vin]*nps))
    # trajsraw = trajs(p, NS, (10,100), 'test', retval=True)
    # sgltrajs = list(single_trajs(trajsraw))
    # plot_trajs(sgltrajs, NS, show=True)
    
    end = time.perf_counter()
    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
