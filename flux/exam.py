import re
import csv
import numpy as np
from os import path
from glob import glob

from dict import Dict

Levels = Dict({})
with open("levels.csv") as f:
    for level in csv.DictReader(f):
        d = Dict(level)
        d.map_(lambda v, k: re.search(r'\s*(.*?)\s*$', v).group(1))
        key = d.pop('key')
        Levels[key] = d

"""
Type declarations 
-----------------

    flux : {
        type    : 'A' | 'V' | 'CSF',
        debit   : [float],
        surface : [float],
        volume  : [float],
        time    : [float],
        nom     : string,
        name    : string
    }

    level   : 'aqueduc'     | 'cid_cereb'   | 'cid_cervi'   
            | 'citer-prep'  | 'cig_cereb'   | 'cig_cervi'
            | 'sinus-d'     | 'coro_sinus_d'| 'coro_sinus_lat_d'
            | 'sinus-s'     | 'coro_sinus_s'| 'coro_sinus_lat_g'
            | 'jugul-d'     | 'verteb-d'    | 'verteb-g'
            | 'v4'          | 'tb'          | 'c2_c3'       

    exam : { level > flux }
"""

def Exam (dirname, *patterns):
    """ Parse all exam files (or only those matching patterns). 

            Exam : str -> { flux } 

        Example:
        --------
        >>> exam = Exam('I', 'coro_sinus*', 'aqueduc')
    """ 
    patterns = ['*'] if not len(patterns) else patterns
    files = []
    fluxes = {}
    for p in patterns:
        files += glob(f'{dirname}/{p}.txt')
    for f in files:
        level = re.search(r'/(.*)\.txt$', f).group(1)
        fluxes[level] = Flux(f)
    return Dict(fluxes)


def Flux (name):
    """ Parse a single exam file.

            Flux : str -> flux

        Example:
        --------
        >>> flux = Flux('I/aqueduc.txt')
    """
    curves = Dict({
        'debit'     : [],
        'surface'   : [],
        'volume'    : [],
        'time'      : []
    })

    with open (name) as f:
        env = None
        for line in f:
            num = re.search(r'-+\s*(-?\d+\.?\d*)', line)
            if  env != None and num != None: 
                curves[env] += [float(num.group(1))]
            elif re.search('DEBIT_mm3/sec', line):
                env = 'debit'
            elif re.search('SURFACE_mm2', line):
                env = 'surface'
            elif re.search('VOLUME_DPL mm3', line):
                env = 'volume'
            elif re.search('Axe des Temps', line):
                env = 'time'
    
    curves.map_(lambda v, k: np.array(v))
    
    key = re.search(r'/(.*)\.txt$', name).group(1)
    curves.update(Levels[key])
    return curves
