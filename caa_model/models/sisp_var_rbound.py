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


class SISPVarRBoundDDM(BaseDDM):
    class Params(NamedTuple):
        c: float  # High confidence boundary
        mu_t: float  # Target drift
        mu_l: float  # Lure drift
        d: float  # Diffusion constant
        tc_bound: float  # Boundary collapse rate
        r_bound_offset: float  # Remember boundary offset
        sigma_r_bound: float  # Variability of the Remember boundary across trials
        z0_t: float  # Target starting point
        z0_l: float  # Lure starting point
        t_post: float  # Post-decision accumulation time
        t0: float  # Non-decision time

    @staticmethod
    def split_params(model_params):
        c_val, mu_t, mu_l, d, tc, r_off, s_r_off, z0_t, z0_l, dT, t0 = model_params
        c_list = [c_val, 0]

        target_params = (c_list, mu_t, d, tc, r_off, s_r_off, z0_t, dT, t0)
        lure_params = (c_list, mu_l, d, tc, r_off, s_r_off, z0_l, dT, t0)

        return target_params, lure_params

    def predicted_proportions(self, params):
        """
        Revised single-accumulator DISP CAA model (Glass, 2026).

        Call twice (once for targets, once for lures) with the appropriate
        parameter values, exactly as the original predicted_proportions() is used.
        For lures, pass mu_r=mu_r_lure, d=d_l, z0=-z0.

        Parameters:
        params: named tuple of model parameters
            c               : confidence boundary (C_High); scalar or 1-D array
            mu_r            : mean recollection drift rate
            d               : diffusion constant
            tc_bound        : boundary collapse rate (tau)
            r_bound_offset  : fixed distance from z0 to remember criterion
            sigma_r_bound   : variability of the remember boundary across trials
            z0              : starting location (recency > 0, novelty < 0)
            t_post          : post-response accumulation interval
            t0              : accumulation start time
        """
        c, mu_r, d, tc_bound, r_bound_offset, sigma_r_bound, z0, t_post, t0 = params
        delta_t = self.config.delta_t
        max_t = self.config.max_t
        nr_tsteps = self.config.nr_tsteps
        nr_ssteps = self.config.nr_ssteps

        # Create confidence bins
        c = pl.array(c, ndmin=1)
        n = len(c)
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

        # Mean position of the Remember criterion
        r_bound_mean = z0 + r_bound_offset

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
        tx = pl.zeros((len(t), len(x)))
        p_old = pl.zeros(pl.shape(t))
        p_new = pl.zeros(pl.shape(t))
        p_rem_conf = pl.zeros((n + 1, pl.size(t)))
        p_know_conf = pl.zeros((n + 1, pl.size(t)))

        # Initialize the probability mass distribution of the first time step
        tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma) * delta_s

        # Iterate through each time step
        for i in range(to_idx, len(t)):
            # Only convolve for steps AFTER the first one
            if i > to_idx:
                tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            # Extract particles that crossed boundaries
            p_pos = tx[i][x >= bound[i]]
            p_old[i] = pl.sum(p_pos)
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])

            # Zero out particles that already crossed the boundary
            tx[i] *= abs(x) < bound[i]

            p_sum = pl.sum(p_pos)
            if p_sum <= EPS:
                continue

            # Find the expected location of the mass that crossed
            x_pos = x[x >= bound[i]]
            crossing_val = pl.dot(p_pos, x_pos) / p_sum

            # --- POST-DECISION BIVARIATE MATH ---
            # 1. Final Accumulated Evidence (X)
            mu_X = crossing_val + mu_r * t_post
            s2_X = (2 * d * t_post) + EPS

            # 2. Variable Boundary (R)
            s2_R = (sigma_r_bound**2) + EPS

            # 3. Difference (Y = X - R). If Y > 0, response is "Remember"
            mu_Y = mu_X - r_bound_mean
            s2_Y = s2_X + s2_R

            # Covariance between X and X-R is simply the variance of X
            cov_XY = s2_X

            # Build the 2D distribution [Final Evidence, Difference]
            mu_mvn = pl.array([mu_X, mu_Y])
            cov_mvn = pl.array([[s2_X, cov_XY], [cov_XY, s2_Y]])

            mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=cov_mvn, allow_singular=True)

            # Slice the 2D plane for confidence and R/K judgments
            for j in range(1, len(clims)):
                c_upper = clims[j - 1]
                c_lower = clims[j]

                # Remember portion: Evidence is in conf bin AND Difference (Y) > 0
                RLL = pl.array([c_lower, 0])
                RUL = pl.array([c_upper, INF_PROXY])

                # Know portion: Evidence is in conf bin AND Difference (Y) < 0
                KLL = pl.array([c_lower, -INF_PROXY])
                KUL = pl.array([c_upper, 0])

                p_rem_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)
                p_know_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)

        return p_rem_conf, p_know_conf, p_new, t


param_bounds = (
    SISPVarRBoundDDM.Params(
        c=0.0,
        mu_t=-2.0,
        mu_l=-2.0,
        d=EPS,
        tc_bound=0.0,
        r_bound_offset=0.0,
        sigma_r_bound=EPS,
        z0_t=-2.0,
        z0_l=-2.0,
        t_post=EPS,
        t0=0.0,
    ),
    SISPVarRBoundDDM.Params(
        c=3.0,
        mu_t=2.0,
        mu_l=2.0,
        d=1.0,
        tc_bound=1.0,
        r_bound_offset=3.0,
        sigma_r_bound=2.0,
        z0_t=2.0,
        z0_l=2.0,
        t_post=2.0,
        t0=1.0,
    ),
)
