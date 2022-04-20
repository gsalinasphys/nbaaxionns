import multiprocessing as mp

import numpy as np

from classes import NeutronStar
from scripts import allhits, outdir, singletrajs


def genconv(eventname):
    # Read trajectories
    simtrajs = np.load(outdir + eventname + '/' + eventname + 'trajs.npy')
    simtrajs = [np.array(simtraj) for simtraj in list(singletrajs(simtrajs))]

    # Finding the points where trajectories hit the conversion surface
    ahits = list(allhits(NS, simtrajs))
    ahits = np.array([hit for subhits in ahits for hit in subhits])

    fnamehits = eventname + 'conversion'
    np.save(outdir + fnamehits, ahits)

# eventnames = ['MCCEBXMO', 'MCBIGZIQ', 'MCYQOGDS', 'MCZLAQEK', 'MCSRACFJ', 'MCGMUOSK', 'MCPCPTTV', 'MCYWLCRK', 'MCFBOPBO', 'MCAEUEWC']
eventnames = ['MCGMUOSK', 'MCPCPTTV', 'MCYWLCRK', 'MCFBOPBO', 'MCAEUEWC']

NS = NeutronStar()

ncores = mp.cpu_count() - 1
with mp.Pool(processes=ncores) as pool:
    pool.starmap(genconv, [(eventname,) for eventname in eventnames])
