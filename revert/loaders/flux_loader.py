import torch
import re
import os

from .flux_reader import Exam, path
from .dict import Dict

class File :

    def __init__(self, dirname):
        self.path = dirname

    def getAll(self, *patterns):
        fluxes = Exam(self.path, *patterns)
        fluxes.map_(lambda fk, k: fk.map_(
            lambda v, _:torch.tensor(v) if type(v) == list else v
        ))
        return fluxes

    def get(self, key): 
        return Exam(self.path, key)[key]

    def sum(self, patterns, channel="debit", sign=1, N=None):
        out = self.getAll(*patterns)\
                  .fmap(lambda f: f[channel])
        if sign == 1:
            out = out.fmap(lambda t: torch.sign(t.mean()) * t) 
        if N != None:
            out = out.fmap(resample(N))
        return out.reduce(lambda v1, v2, _: v1 + v2)


    def volumes(self, level="cervical", N=None):
        
        def vol(*patterns):
            return self.sum(patterns, 'volume')
        
        if re.search(r'cervi', level) != None:
            art = vol('ci*_cervi', 'verteb*')
            ven = vol('jugul*')
        else: 
            art = vol('ci*_cereb', 'tb')
            ven = vol('sinus-d', 'sinus-s')

        csf = - vol('c2_c3')

        V = Dict({"art": art, "ven": ven, "csf": csf})

        alpha = V.art[-1] / V.ven[-1] 
        V.blood = V.art - alpha * V.ven
        V.ic = V.blood - V.csf
        return V

    def fluxes(self, level="cervical", N=None):
        
        def debit (*patterns): 
            return self.sum(patterns, 'debit')
            
        if re.search(r'cervi', level) != None:
            art = debit('ci*_cervi', 'verteb*')
            ven = debit('jugul*')
        else: 
            art = debit('ci*_cereb', 'tb')
            ven = debit('sinus-d', 'sinus-s')

        csf = debit('c2_c3')
        csf = csf * torch.sign(csf @ art)

        return Dict({"art": art, "ven": ven, "csf": csf})

    def age(self): 
        def year (line): 
            match = re.search(r'\d{4}', line)
            return int(match.group(0)) if match else 0

        info = f"{self.path}/general-informations.txt"
        try:
            with open(path(info)) as f:
                l = f.readlines()
                return year(l[3]) - year(l[4])
        except:
            return -1
