from revert.transforms import shuffle_all, shuffle_two, unshuffle

import test
import torch

N, Nc, Npts = 20, 6, 32

class TestShift(test.TestCase):

    # Test with that type of list : [[0...32], [0...32], ... ,[0...32]]
    def test_unshuffle_two(self):
        x = torch.randn([N, Nc, Npts]) 
        one, y = shuffle_two(x)
        result = unshuffle(one,y)
        expect = x
        self.assertClose(expect, result)
            
    # Test with that type of list : [[0...0], [1...1], ... ,[5...5]]

    def test_unshuffle_all(self):
        x2 = torch.randn([N, Nc, Npts])
        two, y2 = shuffle_all(x2)
        result = unshuffle(two, y2)
        expect = x2
        self.assertClose(expect, result)
        
    # Test with [Nc, Npts] tensor
    
    def test_unshuffle_all(self):
        x = torch.randn([Nc, Npts])
        one, y = shuffle_all(x)
        result = unshuffle(one, y)
        expect = x
        self.assertClose(expect, result)

