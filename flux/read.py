import re
import csv
from os import path
from glob import glob

from .dict import Dict

"""
    This module provides basic utilities to parse raw exam directories. 

    Keep this file for text and regex parsing only. 

    Types:
    ------

    flux : {
        type    : 'A' | 'V' | 'CSF',
        debit   : [float],
        surface : [float],
        volume  : [float],
        time    : [float],
        nom     : string,
        name    : string
    }

    level   : 'aqueduc'                 | 'v4'      
            | 'citer-prep'              | 'c2_c3'
            -   -   -   -   -   -   -   -   -   -   -   -
            | 'ci{d,g}_cereb'           | 'verteb-{d,g}'  
            | 'ci{d,g}_cervi'           | 'tb'
            -   -   -   -   -   -   -   -   -   -   -   -   
            | 'sinus-s'                 | 'coro_sinus_s'
            | 'sinus-d'                 | 'coro_sinus_d'
            | 'coro_sinus_lat_{d,g}'    | 'jugul-{d,g}'   


    exam : str -> { level > flux }
"""

Levels = Dict({})
with open(path.join(path.dirname(__file__), "levels.csv")) as f:
    for level in csv.DictReader(f):
        d = Dict(level)
        d.map_(lambda v, k: re.search(r'\s*(.*?)\s*$', v).group(1))
        key = d.pop('key')
        Levels[key] = d


def Exam (dirname, *patterns):
    """ Parse all exam files (or only those matching patterns). 

        Example:
        --------
        >>> exam = Exam('I', 'sinus*', 'aqueduc')
    """
    if dirname[0] != "/":
        dirname = path.join(path.dirname(__file__), 'data', dirname)
    patterns = ['*'] if not len(patterns) else patterns
    files = []
    fluxes = {}
    for p in patterns:
        files += glob(f'{dirname}/{p}.txt')
    for f in files:
        level = re.search(r'/([a-z|0-9|_|-]*)\.txt$', f).group(1)
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

    #key = re.search(r'/([a-z|_|-|0-9]*)\.txt$', name).group(1)
    #curves.update(Levels[key])
    return curves
