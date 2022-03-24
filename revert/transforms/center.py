import torch
from math import factorial as fact
from .transform import Transform

from math      import pi


def unshift(self, xs):
    N  = xs.shape[1]
    Fs = fft(xs, dim=1)
    F1s = Fs[:,1]
    phi = F1s.angle().mean()
    rot = (torch.exp(torch.tensor(2j * pi * phi)) 
        *  F1s.conj() / F1s.abs())
    return ifft(rot * Fs)


class Center (Transform):

    def __init__(self, order=0):
        self.order = order

    def __call__(self, t):
        dj_t = t
        means = []
        N = t.shape[0]
        for j in range(self.order + 1): 
            means += [dj_t.mean()]
            dj_t = torch.diff(t) * N
        q = self.basis(self.order, N)
        delta = sum(mj * q[j] for j, mj in enumerate(means))
        return t - delta

    @classmethod
    def diff(cls, t):
        return t.shape[0] * torch.cat(
            [torch.diff(t), torch.tensor([t[0] - t[-1]])]
        )

    @classmethod
    def basis(cls, order=0, N=5):
        x = torch.linspace(0, 1, N)
        xs = torch.stack([x ** j for j in range(order + 1)])
        cs = torch.stack(cls.coeffs(order))
        return torch.matmul(cs, xs)

    @classmethod
    def coeffs(cls, order=0):
        k = order
        if k == 0:
            return [torch.tensor([1.])]
        q = cls.coeffs(k - 1)
        lower = [q[j] * (- 1. / fact(k + 1 - j)) for j in range(k)]
        lower = torch.sum(torch.stack(lower), dim=[0])
        return [torch.cat([q[j], torch.tensor([0.])]) for j in range(k)]\
            + [torch.cat([lower, torch.tensor([1. / fact(k)])])]
