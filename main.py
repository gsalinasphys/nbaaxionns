import time

import matplotlib.pyplot as plt
import numpy as np

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (energy, grav_en, mag, metropolis, min_approach, randdir,
                     randdirs, rc, rdistr, rm_far, rm_inds, trajAC, trajs,
                     update_ps)


def main() -> None:

    r_in, v_in = [6e3, 1e14, 0], [0, -200., 0]
    nps = 100
    MC = AxionMiniclusterNFW(np.array(r_in), np.array(v_in))
    pMC = Particles(MC.rCM.reshape((1,3)), MC.vCM.reshape((1,3)))
        
    NS = NeutronStar()
    
    # samples = list(metropolis(MC, rdistr, 1_000_000))
    # plt.hist(np.abs(samples), bins=np.linspace(0,0.2,1000))
    # plt.show()
    
    start = time.perf_counter()
    
    print(trajAC(pMC, NS, 100.))
    
    # p = Particles(np.array([r_in]*nps), np.array([v_in]*nps))
    # trajs(p, NS, (10,100), 'test')
    
    end = time.perf_counter()
    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
