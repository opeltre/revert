import torch
import torch.nn as nn

from .module import Module


def cross_correlation (ya, yb):
    """ Cross-correlation of N_batch x N tensors. """
    ya, yb = ya - ya.mean([0]), yb - yb.mean([0])
    yab = ya.T @ yb
    return yab / (ya.norm(dim=0)[:,None] * yb.norm(dim=0))


class Twins (Module):
    """
    Siamese networks sharing weights and structure.

    Given a model mapping shapes [n_batch, *ns] -> [n_batch, *ms],
    the twins will act on shapes as:

        [n_batch, 2, *ns] -> [n_batch, 2, *ms]
    """

    def __init__(self, model, diag=2):
        """ Create twins from a model. """
        super().__init__()
        self.model  = model

    def forward (self, x):
        """ Apply twins to a [n_batch, 2, *ns] tensor. """
        xa, xb = x[:,0], x[:,1]
        ya, yb = self.model(xa), self.model(xb)
        return torch.stack([ya, yb], dim=1)

    def xcorr (self, y):
        """ Cross correlation matrix of twin outputs. """
        return cross_correlation(y[:,0], y[:,1])

    def xcorr_on (self, x):
        """ Cross correlation matrix of outputs from inputs. """
        return self.xcorr(self(x))


class BarlowTwins (Twins):
    """
    Barlow twins with cross-correlation loss.

    See Zbontar et al. (2021)
    => https://arxiv.org/abs/2103.03230
    """

    def __init__(self, model, diag=2):
        """ Create twins with chosen diagonal loss coefficient. """
        self.diag  = diag
        super().__init__(model)

    def loss (self, y):
        """
        Unsupervised Barlow twins loss on outputs.

        Returns the MSE between the cross correlation
        and identity matrices, with diagonal terms
        weighted by `twins.diag`.
        """
        n_out = y.shape[-1]
        C = self.xcorr(y)
        I = torch.eye(n_out, device=C.device)
        w = self.diag
        loss_mask = 1 + (w - 1) * I
        return torch.sum(((C - I) * loss_mask) ** 2) / (n_out ** 2)


class VICReg (Twins):
    """
    Twins with Variance/Invariance/Covariance Regularization loss.

    See Bardes, Ponce, LeCun (2022)
    => https://arxiv.org/abs/2105.04906
    """

    def __init__(self, model, coeffs=(1, 1, .04)):
        """ Create twins with chosen loss coefficients. """
        self.coeffs = (coeffs if isinstance(coeffs, torch.Tensor)
                              else torch.tensor(coeffs))
        super().__init__(model)

    def loss (self, y):
        """
        VICReg loss on outputs, sum of three weighted terms.

            - v(y) = avg_i {max(0, 1 - stdev(y_i))}

            - i(y) = sum_i MSE(y_i, y'_i)

            - c(y) = sum_{i!=j} cross_corr(y_i, y'_j)

        Where y, y' are twin outputs and i, j denote latent space dimensions.
        """
        losses = torch.stack([self.loss_v(y), self.loss_i(y), self.loss_c(y)])
        coeffs = self.coeffs.to(losses.device)
        return (coeffs * losses).sum()

    def loss_v(self, y, eps=1e-6):
        """ Variance regularisation loss. """
        dim = y.shape[-1]
        var = y.var(dim=0)
        dev = torch.sqrt(var + eps).flatten()
        return torch.max(1 - dev, torch.tensor(0)).sum() / dim

    def loss_i(self, y):
        """ Invariance criterion. """
        n_batch = y.shape[0]
        return ((y[:,0] - y[:,1])**2).sum() / n_batch

    def loss_c(self, y):
        """ Covariance criterion, preventing redundancies. """
        dim  = y.shape[-1]
        corr = self.xcorr(y)
        mask = 1 - torch.eye(y.shape[-1], device=y.device)
        return ((mask * corr)**2).sum() / (2 * dim)
