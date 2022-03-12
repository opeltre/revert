from revert.transforms import bandpass, lowpass

import test
import torch

from torch.fft import rfft, irfft
from math import pi 

fs = 100
# 1 Hz sampled at 100Hz for 12s
t  = torch.linspace(0, 12 * 2 * pi, 1200)
x1 = torch.sin(10 * t)
x2 = torch.sin(20 * t)
sp = torch.linspace(0, 50, 601)

bp1 = bandpass(5, 15, 100, N=601)
bp2 = bandpass(18, 22, 100, N=601)

Fx1 = rfft(x1)
Fx2 = rfft(x2)
Fx1 /= rfft(x1).abs().max()
Fx2 /= rfft(x2).abs().max()

class TestBandpass (test.TestCase): 

    def test_bandpass(self):
        result = bp1(x1 + x2)[100:-100]
        expect = x1[100:-100]
        self.assertClose(expect, result, tol=1e-1)
        result = bp2(x1 + x2)[100:-100]
        expect = x2[100:-100]
        self.assertClose(expect, result, tol=1e-1)

    def test_bandpass_resampled(self):
        T = torch.linspace(0, 32 * 2 * pi, 32 * 100)
        y1 = torch.sin(10 * T)
        y2 = torch.sin(20 * T)
        z1 = bp1(y1 + y2)
        z2 = bp2(y1 + y2)
        self.assertClose(y1[100:-100], z1[100:-100], tol=1e-1)
        self.assertClose(y2[200:-200], z2[200:-200], tol=1e-1)
        
