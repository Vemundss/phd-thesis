import jax
import jax.numpy as jnp
from jax import jit

class CANN_burak:
    # CAN based on OG paper (burak 2009)
    def __init__(self, Ng=4096, l=2, tau=1e-2, dt=5e-4, alpha=0.10315, lambda_net=13):
        """
        Initialise CANN of burak

        Parameters
        ----------
        Ng : int
            Number of recurrent nodes
        l : int
            Step size of neural sheet
        tau : float
            Recurrent network time constant
        dt : float
            Discrete simulation time steps
        alpha : float
            Speed coupling factor. In other words, determines speed on neural
            sheet
        lambda_net : int
            Difference of Gaussian tuning parameter. Determines their collective
            width. This also determines spatial frequency of the formed lattice
        """
        self.nl = int(jnp.sqrt(Ng))

        # Create 2D neural sheet (just imagine: topographically arranged cells)
        # shape: (n,n,2)
        self.sheet = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.nl, dtype=jnp.float32) - self.nl / 2,
                jnp.arange(self.nl, dtype=jnp.float32) - self.nl / 2,
            ),
            axis=-1,
        )

        self.dt, self.tau = dt, tau
        self.tc = self.dt / self.tau  # time constant ;)
        self.alpha = alpha
        self.l = l
        self.beta = 3 / lambda_net ** 2
        self.gamma = 1.05 * self.beta

        self.shift = self.init_shifts()
        self.wr = self.init_periodic_recurrent_weights(self.sheet).T  # (nl, nl) ##############

    def relu(self, x):
        return jax.nn.relu(x)

    def init_periodic_recurrent_weights(self, sheet):
        ravel_sheet = jnp.reshape(sheet, (-1, 2))  # (64, 64, 2) -> (64**2, 2)
        ravel_shift = jnp.reshape(self.shift, (-1, 2))
        # compute 'distances'
        d = jnp.abs(ravel_sheet[:, None] - (ravel_sheet + self.l * ravel_shift)[None])
        # compute periodic distances
        #periodic_d = jnp.minimum(d, self.nl - d)  # nl is width of box
        periodic_d = d  # nl is width of box

        papa = jnp.sum(periodic_d ** 2, axis=-1)  # ran out of names
        w0 = jnp.exp(-self.gamma * papa) - jnp.exp(-self.beta * papa)  # DoG
        return w0.reshape((self.nl ** 2, self.nl ** 2))

    def init_shifts(self):
        # shifts are north, south, east and west
        # modify mod2 approach to get each cardinal direction
        p = self.sheet[..., 0]
        q = self.sheet[..., 1]

        a = (p + 1) % 2 * (-1) ** q
        b = p % 2 * (-1) ** q

        shift = jnp.stack((a, b), axis=-1)
        shift = jnp.transpose(shift, (1, 0, -1))
        return shift

@jit
def forward_jit(h, v, wr, shift, alpha, tc):
    recurrence = h@wr#jnp.tensordot(h, wr, axes=(-1, -1))
    velocity_inputs = 1 + alpha * shift@v

    # continuous time rnn
    u = h + tc * (jax.nn.relu(recurrence + velocity_inputs) - h)
    return u