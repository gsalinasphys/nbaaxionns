import multiprocessing as mp
import os
import time

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi']= 300
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from classes import AxionMiniclusterNFW, NeutronStar, Particles
from scripts import (Msun, allhits, cylmax, gag, id_gen, joinnpys, ma,
                     massincyl, outdir, plot_hits, plot_trajs, readme, roche,
                     selectrvs, singletrajs, trajAC, trajs)


# Separate __repr__ functions, as Numba classes do not allow for them
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
Mass:                       {O.mass} x 10^-10 M_Sun
Truncation Radius:          {O.rtrunc():.2e} km
Delta:                      {O.delta}
Concentration:              {O.c}
Velocity dispersion:        {'Maxwell-Boltzmann' if O.vdisptype else 'None'}
"""
        )
    elif isinstance(O, tuple):
        return (
f"""---------- Encounter data ----------
Initial clump position:     [{O[0][0]:.2e}, {O[0][1]:.2e}, {O[0][2]:.2e}] km
Initial clump velocity:     [{O[1][0]:.2e}, {O[1][1]:.2e}, {O[1][2]:.2e}] km/s
Impact parameter:           {O[2]:.2e} km
Roche radius:               {O[3]:.2e} km
Position at Roche radius:   [{O[4][0]:.2e}, {O[4][1]:.2e}, {O[4][2]:.2e}] km
Velocity at Roche radius:   [{O[5][0]:.2e}, {O[5][1]:.2e}, {O[5][2]:.2e}] km/s
Sampling cylinder:          R = {O[6]:.2e} km, z = {O[7]:.2e} + [{O[8][0]:.2e}, {O[8][1]:.2e}] km
Axions per trajectory:      {O[9]:.2e}
"""
        )


def run(nps: int, b: float, vin: np.ndarray, 
             ACparams: tuple = (1, np.empty((0,3)), np.empty((0,3)), 1., 1.55, 100., 1), lbounds: tuple = (-1., 1.),
             NSparams: tuple = (1.,10.,1.,np.array([0.,0.,1.]),1.,0.,0.),
             rprecision: float = 5e-2, padding: float = 10.,
             fnametrajs: str = None, fnamehits: str = None) -> np.ndarray:
    if ACparams[0]:
        rCM, vCM, ACmass, delta, c, vdisptype = ACparams[1:]
        AC = AxionMiniclusterNFW(rCM=rCM, vCM=vCM, mass=ACmass, delta=delta, c=c, vdisptype=vdisptype)
    
    NSmass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=NSmass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    
    rvsinps, ndrawn = selectrvs(AC, NS, nps, lbounds)
    ps = Particles(rvsinps[0], rvsinps[1])
    
    simtrajs = trajs(ps, NS, (NS.radius, NS.rcmax(padding)), rprecision, fnametrajs)
    simtrajs = [np.array(simtraj) for simtraj in list(singletrajs(simtrajs))]
    
    ahits = list(allhits(NS, simtrajs))
    ahits = np.array([hit for subhits in ahits for hit in subhits])
    
    if fnamehits is not None:
        np.save(outdir + fnamehits, ahits)
        
    return ahits, ndrawn

def main() -> None:
    nps, b, vin = 160, 0.2, 200.
    
    ACparams = (1, 1., 1.55, 100., 1)
    if ACparams[0]:
        ACmass, delta, c, vdisptype = ACparams[1:]
        AC = AxionMiniclusterNFW(mass=ACmass, delta=delta, c=c, vdisptype=vdisptype)
        lbounds = (-1., 1.)
    
    NSparams = (1.,10.,1.,np.array([0.,0.,1.]),1.,0.,0.)
    NSmass, radius, T, axis, B0, misalign, psi0 = NSparams
    NS = NeutronStar(mass=NSmass, radius=radius, T=T, axis=axis, B0=B0, misalign=misalign, psi0=psi0)
    
    rprecision, padding = 5e-2, 10.
    
    eventname = ('MC' if ACparams[0] else 'AS') + id_gen()
    savetrajs, savehits = True, True
    os.makedirs(outdir + eventname, exist_ok=True)
    readme(eventname,
           f"""Event name:                 {eventname}
Axion mass:                 {ma} x 10^-5 eV
Axion-photon coupling:      {gag} x 10^-14 GeV-1
\n""")
    readme(eventname, repr(NS) + '\n')
    readme(eventname, repr(AC) + '\n')
    
    rvsin = np.array([b*AC.rtrunc(), 0., 1e16]), np.array([0.,0.,-vin])
    pAC = Particles(np.array([rvsin[0]]), np.array([rvsin[1]]))
    trajAC(pAC, NS, roche(AC, NS), rprecision)
    AC.rCM, AC.vCM = pAC.positions[0], pAC.velocities[0]
    ACparams = (ACparams[0],) + (AC.rCM, AC.vCM) + ACparams[1:]
    
    ncores = mp.cpu_count() - 1
    with mp.Pool(processes=ncores) as pool:
        result = pool.starmap(run, [(nps, b, vin, ACparams, lbounds,
                                     NSparams, rprecision, padding,
                                     eventname+'/'+eventname+f'_{ii}'+'trajs' if savetrajs else None, 
                                     eventname+'/'+eventname+f'_{ii}'+'conversion' if savehits is not None else None)
                                    for ii in range(ncores)])
    
    joinnpys(eventname)
    
    trajs = np.load(''.join([outdir, eventname, '/', eventname, 'trajs.npy']))
    sgltrajs = list(singletrajs(trajs))
    plot_trajs(sgltrajs, NS, fname=''.join([outdir, eventname, '/', eventname, 'trajs']))
    
    result, ndrawn = np.concatenate([rst[0] for rst in result]), sum([rst[1] for rst in result])
    masscyl = massincyl(AC, ((0., cylmax(AC, NS)), lbounds))
    axspertraj = masscyl*1e-10*Msun/(1e-5*ma*ndrawn)
        
    plot_hits(result, eventname)
    
    towrite = (rvsin[0], rvsin[1], b*AC.rtrunc(), roche(AC, NS), AC.rCM, AC.vCM,
               cylmax(AC,NS)*AC.rtrunc(), AC.rCM[2], AC.rtrunc()*np.array(lbounds), axspertraj
               )
    readme(eventname, repr(towrite))


if __name__ == '__main__':
    start = time.perf_counter()
    
    main()
    
    end = time.perf_counter()
    print("Run time: ", np.round(end - start, 2))
