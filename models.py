import torch
import numpy as np
import copy

from methods import HexagonalGCs


class JacobianCI(HexagonalGCs):
    def __init__(self, scale=None, p_magnitude=0, lr=1e-3, **kwargs):
        super(JacobianCI, self).__init__(**kwargs)
        self.p_magnitude = p_magnitude
        # scale of similitude
        self.set_scale(scale)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def set_scale(self, scale=None):
        if scale is None:
            # conformally isometric scaling LAW
            A = 2/9
            scale = 3 * np.pi**2 * self.ncells * A**2
        self.scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=self.dtype), requires_grad=True
        )
        return self.scale

    def loss_fn(self, r):
        dp, _ = (
            self.jitter(self.phases.shape[0], magnitude=self.p_magnitude)
            if self.p_magnitude
            else (None, None)
        )
        # dp = torch.normal(0, self.p_magnitude, size=(r.shape[0], self.ncells), dtype=self.dtype)
        J = self.jacobian(r, dp)
        # (nsamples,2,2)
        metric_tensor = self.metric_tensor(J)
        diag_elems = torch.diagonal(metric_tensor, dim1=-2, dim2=-1)
        lower_triangular_elems = torch.tril(metric_tensor, diagonal=-1)
        loss = torch.sum((diag_elems - self.scale) ** 2, dim=-1) + 2 * torch.sum(
            lower_triangular_elems**2, dim=(-2, -1)
        )
        return torch.mean(loss)


class Torus:
    def __init__(self, R=1, r=0.5, alpha=0, beta=0, n=100, a=1, b=1):
        self.R = R
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.a = a
        self.b = b


def torus(R=1, r=0.5, alpha=0, beta=0, n=100, a=1, b=1):
    """
    Generate a torus with radius R and inner radius r
    R > r => ring-torus (standard), R = r => Horn-torus, R < r => Spindel-torus

    params:
        R: radius of outer circle
        r: radius of inner circle
        n: square root of number of points on torus
        alpha: outer circle twist parameter wrt. inner circle. (twisted torus)
        beta: inner circle twist parameter wrt. outer circle. (unknown)
    """
    theta = np.linspace(-a * np.pi, a * np.pi, n)  # +np.pi/2
    phi = np.linspace(-b * np.pi, b * np.pi, n)
    x = (R + r * np.cos(theta[None] - alpha * phi[:, None])) * np.cos(
        phi[:, None] - beta * theta[None]
    )
    y = (R + r * np.cos(theta[None] - alpha * phi[:, None])) * np.sin(
        phi[:, None] - beta * theta[None]
    )
    z = r * np.sin(theta[None] - alpha * phi[:, None])
    coords = np.array([x, y, z]).T
    return coords