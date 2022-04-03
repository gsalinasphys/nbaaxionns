import time

import numpy as np

from classes import NeutronStar, Particles
from scripts import update_ps


def main():
    start = time.perf_counter()

    r_in, v_in = [6e3, 1e14, 0], [0, -200., 0]
    p = Particles(np.array([r_in]*10), np.array([v_in]*10))

    NS = NeutronStar()

    [update_ps(p, NS) for _ in range(60000)]

    end = time.perf_counter()

    print("Run time: ", np.round(end - start, 2))

if __name__ == '__main__':
    main()
