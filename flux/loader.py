import torch

from .read import Exam 
from .dict import Dict

class File :
    def __init__(self, dirname):
        self.path = dirname

    def getAll(self, *patterns):
        fluxes = Exam(self.path, *patterns)
        fluxes.map_(
            lambda v, k: np.array(v) if type(v) == list else v
        )
        return fluxes

    def get(self, key): 
        return Exam(self.path, key)[key]

    def blood_volume_change(self, level="cervical"):
        if re.search(r'cervi', level) != None:
            art = self.get('ci*_cervi', 'verteb*')
            ven = self.get('jugul*')
        else: 
            art = self.get('ci*_cereb', 'tb')
            ven = self.get('sinus-d', 'sinus-s')
        return art.volume - ven.volume


    def volume_change(self, level="cervical"):
        csf = self.get('c2_c3')
        blood = self.blood_volume_change(level)
        return blood - csf.volume


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
