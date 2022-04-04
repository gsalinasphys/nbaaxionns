import time

import numpy as np

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (energy, grav_en, mag, min_approach, randdir, randdirs,
                     rm_far, rm_inds, trajs, update_ps)


def main() -> None:
    start = time.perf_counter()

    r_in, v_in = [6e3, 1e14, 0], [0, -200., 0]
    p = Particles(np.array([r_in]*100), np.array([v_in]*100))
    
    NS = NeutronStar()
    
    MC = AxionMiniclusterNFW(np.array(r_in), np.array(v_in))
    
    pos = np.array(r_in) + np.array([10000., 1., 1.])
    print(MC.vsdisp(np.array([pos]*10000)))
            
    end = time.perf_counter()

    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
