import time

import numpy as np

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (draw_rs, energy, grav_en, mag, min_approach, randdir,
                     randdirs, randinsphere, rm_far, rm_inds, trajs, update_ps)


def main() -> None:
    start = time.perf_counter()

    r_in, v_in = [6e3, 1e14, 0], [0, -200., 0]
    p = Particles(np.array([r_in]*100), np.array([v_in]*100))
    
    NS = NeutronStar()
    
    MC = AxionMiniclusterNFW(np.array(r_in), np.array(v_in))
    
    draw_rs(MC, 100)
            
    end = time.perf_counter()

    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
