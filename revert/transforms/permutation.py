import torch

class Permutation:

    def __init__(self, sigma):
        self.idx = sigma

    def __call__(self, input, dim=0):
        return input.index_select(dim, self.idx)

    def __matmul__(self, other):
        return Permutation(other.idx.index_select(0, self.idx))
    
    @classmethod
    def rand(cls, n, device=None):
        return cls(torch.randperm(n, device=device))
    
    @classmethod
    def swap(cls, n, i, j, device=None):
        perm = torch.arange(0, n, device=device)
        perm[j] = i
        perm[i] = j
        return cls(perm)

    def __repr__(self):
        return (str(self.idx).replace("tensor", "Permutation")
                             .replace("      ", "           "))
