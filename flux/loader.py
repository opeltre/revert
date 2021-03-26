import torch
import re
import sig

from .read import Exam 
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

    def volumes(self, level="cervical", N=None):
        
        def vol(*patterns):
            return self.getAll(*patterns)\
                .fmap(lambda f: f.volume[:-1])\
                .fmap(lambda t: sig.resample(t, N) if N else t)\
                .reduce(lambda v1, v2, _: v1 + v2)
        
        if re.search(r'cervi', level) != None:
            art = vol('ci*_cervi', 'verteb*')
            ven = - vol('jugul*')
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
            return self.getAll(*patterns)\
                .fmap(lambda f: f.debit)\
                .fmap(lambda t: sig.resample(t, N) if N else t)\
                .reduce(lambda v1, v2, _: v1 + v2)
            
        if re.search(r'cervi', level) != None:
            art = debit('ci*_cervi', 'verteb*')
            ven = - debit('jugul*')
        else: 
            art = debit('ci*_cereb', 'tb')
            ven = debit('sinus-d', 'sinus-s')
        csf = - debit('c2_c3')

        return Dict({"art": art, "ven": ven, "csf": csf})


class Flux (Dict):

    def resample(self, N): 
        return self

    def fft(self): 
        return np.fft(self.debit)

    def __add__(self, other):
        return Flux({
            'time'      : self.time,
            'debit'     : self.debit + other.debit,
            'volume'    : self.volume + other.volume,
            'surface'   : self.surface + other.surface
        })
