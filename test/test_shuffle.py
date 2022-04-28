from revert.transforms import shuffle_all, shuffle_two, unshuffle

import test
import torch

# shuffle_all() and shuffle_two
N = 500
x = torch.arange(32).repeat(6).repeat(N).view([N, 6 ,32])

x2 = []
for i in range(N) :  
    x2.append([])
    for j in range(6)  :
        x2[i].append([j for _ in range(32)])
x2 = torch.tensor(x2)

class TestShift(test.TestCase):

    # Test with that type of list : [[0...32], [0...32], ... ,[0...32]]
    def test_unshuffe_two(self):
        one, y = shuffle_two(x)
        result = unshuffle(one,y)
        expect = x
        self.assertClose(expect, result)

        
            
    # Test with that type of list : [[0...0], [1...1], ... ,[5...5]]

    def test_unshuffle_all(self):
        two, y2 = shuffle_all(x2)
        result = unshuffle(two, y2)
        expect = x2
        self.assertClose(expect, result)
        
    # Test with that type of list : [[0...32], [0...32], ... ,[0...32]] with N = 1
    
    def test_unshuffle_all(self):
        one, y = shuffle_all(x[0])
        result = unshuffle(one, y)
        expect = x[0]
        self.assertClose(expect, result)

