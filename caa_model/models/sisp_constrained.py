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


class SISPConstrainedDDM(BaseDDM):
    class Params(NamedTuple):
        c: float  # High confidence boundary
        mu_t: float  # Target drift
        mu_l: float  # Lure drift
        d: float  # Diffusion constant
        tc_bound: float  # Boundary collapse rate
        r_bound_offset: float  # Remember boundary offset
        z0_t: float  # Target starting point
        z0_l: float  # Lure starting point
        t_post: float  # Post-decision accumulation time
        sigma_z0: float  # Starting position variability
        t0: float  # Non-decision time

    @staticmethod
    def split_params(model_params):
        c_val, mu_t, mu_l, d, tc, r_off, z0_t, z0_l, dT, s_z0, t0 = model_params
        c_list = [c_val, 0]

        target_params = (c_list, mu_t, d, tc, r_off, z0_t, dT, s_z0, t0)
        lure_params = (c_list, mu_l, d, tc, r_off, z0_l, dT, s_z0, t0)

        return target_params, lure_params

    def predicted_proportions(self, params):
        """
        Revised single-accumulator DISP CAA model (Glass, 2026).

        Call twice (once for targets, once for lures) with the appropriate
        parameter values, exactly as the original predicted_proportions() is used.
        For lures, pass mu_r=mu_r_lure, d=d_l, z0=-z0.

        Parameters:
        params: named tuple of model parameters
            c              : confidence boundary (C_High); scalar or 1-D array
            mu_r           : mean recollection drift rate
            d              : diffusion constant
            tc_bound       : boundary collapse rate (tau)
            r_bound_offset : fixed distance from z0 to remember criterion
            z0             : starting location (recency > 0, novelty < 0)
            t_post         : post-response accumulation interval
            t0             : accumulation start time
            sigma_z0       : variability in percieved familiarity across trials
        """
        c, mu_r, d, tc_bound, r_bound_offset, z0, t_post, sigma_z0, t0 = params
        delta_t = self.config.delta_t
        max_t = self.config.max_t
        nr_tsteps = self.config.nr_tsteps
        nr_ssteps = self.config.nr_ssteps

        # Create confidence bins
        c = pl.array(c, ndmin=1)
        n = len(c)
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

        # Set remember criterion position
        r_bound = z0 + r_bound_offset

        # Standard deviation and mean drift of the accumulator per-step
        sigma = pl.sqrt(2 * d * delta_t)
        mu = mu_r * delta_t

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

        # Initializing output arrays
        tx = pl.zeros((len(t), len(x)))  # RT distribution at each time point
        p_old = pl.zeros(pl.shape(t))  # Probability mass hitting upper collapsing bound at each time point
        p_new = pl.zeros(pl.shape(t))  # Probability mass hitting lower collapsing bound at each time point
        p_rem_conf = pl.zeros((n + 1, pl.size(t)))  # Yes responses that are remember (cross r_bound)
        p_know_conf = pl.zeros((n + 1, pl.size(t)))  # Yes responses that are known (does not cross r_bound)

        # Initialize the probability mass distribution of the first time step
        sigma_init = pl.sqrt(sigma**2 + sigma_z0**2)
        tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma_init) * delta_s

        # Iterate through each time step
        for i in range(to_idx, len(t)):
            # Only convolve for steps AFTER the first one
            if i > to_idx:
                # Uses convolution to advance the probability mass distribution by one step
                tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            # Extract particles that crossed boundaries
            p_pos = tx[i][x >= bound[i]]
            p_old[i] = pl.sum(p_pos)
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])

            # Zero out particles that already crossed the boundary
            tx[i] *= abs(x) < bound[i]

            p_sum = pl.sum(p_pos)
            if p_sum <= EPS:
                # No particles crossed the bound
                continue

            # Find the expected location of the mass that crossed
            x_pos = x[x >= bound[i]]
            crossing_val = pl.dot(p_pos, x_pos) / p_sum

            # Mean and SD of accumulator after deltaT
            mu_delta = crossing_val + mu_r * t_post
            std_delta = pl.sqrt(2 * d * t_post)

            # Distribution of accumulator after deltaT
            dist = stats.norm(loc=mu_delta, scale=std_delta)

            for j in range(1, len(clims)):
                c_upper = clims[j - 1]
                c_lower = clims[j]

                # Remember portion
                rem_lower = max(c_lower, r_bound)
                rem_upper = c_upper
                if rem_upper > rem_lower:
                    p_rem_conf[j - 1, i] = p_old[i] * (dist.cdf(rem_upper) - dist.cdf(rem_lower))

                # Know portion
                know_lower = c_lower
                know_upper = min(c_upper, r_bound)
                if know_upper > know_lower:
                    p_know_conf[j - 1, i] = p_old[i] * (dist.cdf(know_upper) - dist.cdf(know_lower))

        return p_rem_conf, p_know_conf, p_new, t


# Parameters obtained from using 10 quantiles
params_est = SISPConstrainedDDM.Params(
    1.7802366345277048,
    1.0732968669582117,
    0.17804995041166943,
    0.5326415575619126,
    0.043589790458315514,
    0.9574460424899932,
    -0.690027580392909,
    -0.565893039286282,
    1.5546604244949078,
    0.12666843828878627,
    0.46706560655302104,
)

param_bounds = (
    SISPConstrainedDDM.Params(
        c=0.0,
        mu_t=0.0,
        mu_l=0.0,
        d=EPS,
        tc_bound=0.0,
        r_bound_offset=0.0,
        z0_t=-2.0,
        z0_l=-2.0,
        t_post=EPS,
        sigma_z0=EPS,
        t0=0.0,
    ),
    SISPConstrainedDDM.Params(
        c=3.0,
        mu_t=2.0,
        mu_l=1.0,
        d=1.0,
        tc_bound=1.0,
        r_bound_offset=3.0,
        z0_t=2.0,
        z0_l=2.0,
        t_post=2.0,
        sigma_z0=EPS,
        t0=1.0,
    ),
)
