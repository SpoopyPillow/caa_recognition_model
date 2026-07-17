# configure for compatibility with Python 3
from __future__ import absolute_import, division, print_function

# standard library imports
from typing import NamedTuple

# scientific library imports
import pylab as pl
from scipy import stats

# local imports
from .base_model import BaseDDM
from ..config import EPS, INF_PROXY
from ..utils.multinomial_funcs import get_rect_prob


class DISPClassicDDM(BaseDDM):
    class Params(NamedTuple):
        c: float
        mu_r: float
        mu_f: float
        d_r: float
        d_f: float
        tc_bound: float
        r_bound: float
        z0: float
        mu_r_new: float
        mu_f_new: float
        t_post: float
        t0: float

    @staticmethod
    def split_params(model_params):
        c, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, mu_r0, mu_f0, t_post, t0 = model_params
        c_list = [c, 0]

        target_params = (c_list, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, t_post, t0)
        lure_params = (c_list, mu_r0, mu_f0, d_r, d_f, tc_bound, r_bound, z0, t_post, t0)

        return target_params, lure_params

    def predicted_proportions(self, params):
        """
        Revised DISP model with staggered recollection time offset.
        """
        c, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, t_post, t0 = params
        delta_t = self.config.delta_t
        max_t = self.config.max_t
        nr_tsteps = self.config.nr_tsteps
        nr_ssteps = self.config.nr_ssteps

        # Create confidence bins
        c = pl.array(c, ndmin=1)
        n = len(c)
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

        # --- Pre-compute per-step constants and grid ---
        # Compute SD and mean drift of the accumulator per-step
        sigma_r = pl.sqrt(2 * d_r * delta_t)
        sigma_f = pl.sqrt(2 * d_f * delta_t)
        sigma = pl.sqrt(sigma_r**2 + sigma_f**2)
        mu = (mu_r + mu_f) * delta_t

        # Compute the correlation for r given r+f
        rho = sigma_r / sigma

        # Create the time axis, where to_idx is the index of the first time step
        t = pl.linspace(delta_t, max_t, nr_tsteps)
        to_idx = pl.argmin((t - t0) ** 2)
        # Bound is the collapsing boundary at each time point
        bound = pl.exp(-tc_bound * pl.clip(t - t0, 0, None))

        # Create the grid for the accumulator
        space_lim = max(bound) + 3 * sigma
        delta_s = 2 * space_lim / nr_ssteps
        x = pl.linspace(-space_lim, space_lim, nr_ssteps)

        # Kernel is the probability mass function for each step
        kernel = stats.norm.pdf(x, mu, sigma) * delta_s
        # FFT to prepare for convolution
        ft_kernel = self._fft(kernel)

        # --- Initializing output arrays ---
        tx = pl.zeros((len(t), len(x)))  # RT distribution at each time point
        p_old = pl.zeros(pl.shape(t))  # Probability mass hitting upper collapsing bound at each time point
        p_new = pl.zeros(pl.shape(t))  # Probability mass hitting lower collapsing bound at each time point
        p_rem_conf = pl.zeros((n + 1, pl.size(t)))  # Yes responses that are remember (cross r_bound)
        p_know_conf = pl.zeros((n + 1, pl.size(t)))  # Yes responses that are known (does not cross r_bound)

        for i in range(to_idx, len(t)):
            t_elapsed = (t[i] - t[to_idx]) + delta_t

            # --- Move the particle ---
            # Initialize or advance the probability mass distribution by one step
            if i == to_idx:
                # Initialize starting normal distribution
                tx[i] = stats.norm.pdf(x, mu + z0, sigma) * delta_s
            else:
                # Convolve previous state with kernel
                tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            # --- Identify particles that cross the bound ---
            p_pos = tx[i][x >= bound[i]]
            p_old[i] = pl.sum(p_pos)
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])

            # Remove from consideration any particles that already hit the bound
            tx[i] *= abs(x) < bound[i]

            p_sum = pl.sum(p_pos)
            if p_sum <= EPS:
                # No particles crossed the bound
                continue

            # Find the expected location of the mass that crossed
            x_pos = x[x >= bound[i]]
            comb_est = pl.dot(p_pos, x_pos) / p_sum

            # --- Statistics of particles that cross the bound ---
            # Compute STD(r) for the current time
            s_r = pl.sqrt(2 * d_r * t_elapsed)
            s_f = pl.sqrt(2 * d_f * t_elapsed)

            # Compute STD[r|(r+f) = bound]
            s_r_cond = s_r * pl.sqrt(1 - rho**2)
            s_f_cond = s_f * pl.sqrt(1 - (sigma_f / sigma) ** 2)

            # Compute E[r|(r+f) > bound]
            mu_r_cond = mu_r * t_elapsed + (comb_est - t_elapsed * (mu_r + mu_f) - z0) * rho**2

            # --- Post-decision bivariate distribution (t_post seconds after decision) ---
            # Build 2D MVN distribution [Recollection, Total Evidence (Recollection + Familiarity)]

            # Expected value of post-decision MVN
            mu_r_delta = mu_r_cond + mu_r * t_post
            mu_comb_delta = (mu_r + mu_f) * t_post + comb_est

            mu_mvn = pl.array([mu_r_delta, mu_comb_delta])

            # Covariance matrix of post-decision MVN
            # Isolated post-decision variance
            s2_r_post = 2 * d_r * t_post
            s2_f_post = 2 * d_f * t_post

            # Recollection variance includes conditional boundary variance + post-decision variance
            s2_r_delta = s_r_cond**2 + s2_r_post
            # Total evidence variance includes only post-decision variance
            s2_comb_delta = s2_r_post + s2_f_post
            # Covariance (between post-decision r and r+f) ONLY depends on post-decision recollection variance
            cov_delta = s2_r_post

            # Inject microscopic variance to prevent SciPy divide-by-zero crashes
            s2_r_delta += EPS
            s2_comb_delta += EPS

            sigma_mvn = pl.array([[s2_r_delta, cov_delta], [cov_delta, s2_comb_delta]])

            # TODO: Batch MVN instead of instantiating new object each time step
            # Build MVN
            mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=sigma_mvn, allow_singular=True)

            # --- Compute conditional probability of each confidence level ---
            for j in range(1, len(clims)):
                # Note: clims contains the bin edges in descending order
                KLL = pl.array([-INF_PROXY, clims[j]])
                KUL = pl.array([r_bound, clims[j - 1]])
                RLL = pl.array([r_bound, clims[j]])
                RUL = pl.array([INF_PROXY, clims[j - 1]])
                p_know_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)
                p_rem_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)

        return p_rem_conf, p_know_conf, p_new, t


params_est = DISPClassicDDM.Params(
    0.9984, 0.002, 0.3035, 0.0037, 0.3736, 0.0585, 0.001, -0.1306, -0.0142, -0.2438, 0.5859, 0.5126
)

param_bounds = [
    DISPClassicDDM.Params(
        c=0.0,
        mu_r=-2.0,
        mu_f=-2.0,
        d_r=EPS,
        d_f=EPS,
        tc_bound=0.05,
        r_bound=0.0,
        z0=-1.0,
        mu_r_new=-2.0,
        mu_f_new=-2.0,
        t_post=EPS,
        t0=0.0,
    ),
    DISPClassicDDM.Params(
        c=1.0,
        mu_r=2.0,
        mu_f=2.0,
        d_r=1.0,
        d_f=1.0,
        tc_bound=1.0,
        r_bound=1.0,
        z0=1.0,
        mu_r_new=2.0,
        mu_f_new=2.0,
        t_post=2.0,
        t0=0.5,
    ),
]
