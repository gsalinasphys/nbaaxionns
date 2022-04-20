import os
import random
import string

import numpy as np

from scripts.globals import outdir


def id_gen(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def joinnpys(dirstr):
    datatrajs, dataconv, tagtrajs, tagconv = [], [], 0, 0
    for file in os.listdir(os.fsencode(outdir + dirstr)):
        filename, ext = os.path.splitext(os.fsdecode(file))
        if ext == ".npy":
            if filename.endswith('trajs'):
                continue
                # databt = np.load(''.join([outdir, dirstr, '/', filename, ext]))
                # databt += np.array([tagtrajs] + [0]*8)
                # tagtrajs = databt[-1][0] + 1
                # datatrajs.append(databt)
                # os.remove(''.join([outdir, dirstr, '/', filename, ext]))
            elif filename.endswith('conversion'):
                databt = np.load(''.join([outdir, dirstr, '/', filename, ext]))
                databt += np.array([tagconv] + [0]*8)
                tagconv = databt[-1][0] + 1
                dataconv.append(databt)
                os.remove(''.join([outdir, dirstr, '/', filename, ext]))
    
    # np.save(''.join([outdir, dirstr, '/', dirstr, 'trajs']), np.concatenate(datatrajs))
    np.save(''.join([outdir, dirstr, '/', dirstr, 'conversion']), np.concatenate(dataconv))
    
def readme(eventname: str, text: str) -> None:
    with open(''.join([outdir, eventname, '/', 'README.txt']), 'a') as f:
        f.write(text)
