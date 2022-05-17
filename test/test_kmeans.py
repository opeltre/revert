import test
import torch

from revert.models import KMeans

n_clusters = 4
dim  = 2
n_it = 3000
eps  = 1e-6

def makeClusters(stdev=0.1, n=100):
    """ Generate n points close to each corner of a square. """
    means = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    noise = stdev * torch.randn([n, 4, 2])
    x = means + noise
    return means, x.view([n * 4, 2])

class TestKMeans (test.TestCase):
   
    @test.skipFit("KMeans", "2d-gaussians centered on square corners")
    def test_fit(self, writer):
        km = KMeans(n_clusters, dim)
        km.writer = writer
        corners, x = makeClusters()
        km.fit([x], epochs=6000, tag="kmeans")
        print(f'\n\t=> centers:\n{km.centers.cpu().detach()}')
        y = km.predict(x).float().view([-1, n_clusters, dim])
        # Expect prediction to be constant across clusters 
        result = y - y.mean([0])
        expect = torch.zeros(y.shape)
        self.assertClose(expect, result)
        # Expect clusters close to square corners 
        corners = corners.float()
        idx = km.predict(corners)
        self.assertClose(km.centers[idx], corners, tol=.2)
        # Expect clusters to match cluster means
        means = x.view([-1, n_clusters, dim]).mean([0])
        self.assertClose(km.centers[idx], means, tol=1e-6)
