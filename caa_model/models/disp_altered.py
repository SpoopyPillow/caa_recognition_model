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


class DISPAltered(BaseDDM):
    """
    Hybrid Continuous Aggregated Accumulation Model
    - Targets: Dual-Accumulator (DISP) using both Recollection and Familiarity.
    - Lures: Single-Accumulator (SISP) with 0.0 drift and a Variable Remember Boundary.
    """
    class Params(NamedTuple):
        c: float              # High confidence boundary
        mu_r_t: float         # Target recollection drift
        mu_f_t: float         # Target familiarity drift
        d_r_t: float          # Target recollection diffusion
        d_f_t: float          # Target familiarity diffusion
        d_l: float            # Lure diffusion (SISP single track)
        tc_bound: float       # Shared boundary collapse rate
        r_bound_offset: float # Distance from z0 to the mean remember boundary
        sigma_r_bound: float  # Variability of the lure remember boundary
        z0_t: float           # Target starting point
        z0_l: float           # Lure starting point
        t_post: float         # Post-decision accumulation
        t0: float             # Non-decision time

    @staticmethod
    def split_params(model_params):
        (c_val, mu_r_t, mu_f_t, d_r_t, d_f_t, d_l,
         tc, r_off, s_r_off, z0_t, z0_l, dT, t0) = model_params

        c_list = [c_val, 0.0]

        # Target uses standard DISP parameters. is_target flag = True.
        target_params = (c_list, mu_r_t, mu_f_t, d_r_t, d_f_t, tc, r_off, s_r_off, z0_t, dT, t0, True)

        # Lure enforces 0.0 drift. d_r parameter takes the d_l value. is_target flag = False.
        lure_params = (c_list, 0.0, 0.0, d_l, 0.0, tc, r_off, s_r_off, z0_l, dT, t0, False)

        return target_params, lure_params

    def predicted_proportions(self, params):
        c, mu_r, mu_f, d_r, d_f, tc_bound, r_bound_offset, sigma_r_bound, z0, t_post, t0, is_target = params

        delta_t = self.config.delta_t
        max_t = self.config.max_t
        nr_tsteps = self.config.nr_tsteps
        nr_ssteps = self.config.nr_ssteps

        c = pl.array(c, ndmin=1)
        n = len(c)
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

        # Total drift and diffusion for the 1D convolution
        d_total = d_r + d_f if is_target else d_r
        mu = (mu_r + mu_f) * delta_t
        sigma = pl.sqrt(2 * d_total * delta_t)

        t = pl.linspace(delta_t, max_t, nr_tsteps)
        to_idx = pl.argmin((t - t0) ** 2)

        bound = pl.exp(-tc_bound * pl.clip(t - t0, 0, None))

        space_lim = max(bound) + 3 * sigma
        delta_s = 2 * space_lim / nr_ssteps
        x = pl.linspace(-space_lim, space_lim, nr_ssteps)

        kernel = stats.norm.pdf(x, mu, sigma) * delta_s
        ft_kernel = self._fft(kernel)

        tx = pl.zeros((len(t), len(x)))
        p_old = pl.zeros(pl.shape(t))
        p_new = pl.zeros(pl.shape(t))
        p_rem_conf = pl.zeros((n + 1, pl.size(t)))
        p_know_conf = pl.zeros((n + 1, pl.size(t)))

        tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma) * delta_s

        for i in range(to_idx, len(t)):
            if i > to_idx:
                tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            p_pos = tx[i][x >= bound[i]]
            p_old[i] = pl.sum(p_pos)
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])

            tx[i] *= abs(x) < bound[i]

            p_sum = pl.sum(p_pos)
            if p_sum <= EPS:
                continue

            x_pos = x[x >= bound[i]]
            crossing_val = pl.dot(p_pos, x_pos) / p_sum

            # --- HYBRID POST-DECISION BIVARIATE MATH ---
            t_drift = max(t[i] - t0, EPS)
            r_bound_mean = z0 + r_bound_offset

            if is_target:
                # 1. DISP Math for Targets[cite: 15]
                # Y = Total Evidence (R + F)
                mu_Y = crossing_val + (mu_r + mu_f) * t_post
                s2_Y = (2 * d_total * t_post) + EPS

                # E[R_cond] and Var(R_cond) at boundary crossing
                delta_Y = crossing_val - z0
                drift_Y = (mu_r + mu_f) * t_drift
                mu_R_cond = z0 + (mu_r * t_drift) + (d_r / d_total) * (delta_Y - drift_Y)
                s2_R_cond = (2 * d_r * t_drift) * (1.0 - (d_r / d_total))

                # Add post-decision accumulation
                mu_R = mu_R_cond + (mu_r * t_post)
                s2_R = s2_R_cond + (2 * d_r * t_post) + EPS

                # Shift X axis so Remember threshold is at 0.0
                mu_X = mu_R - r_bound_mean
                s2_X = s2_R

                cov_XY = (2 * d_r * t_post) + EPS

            else:
                # 2. SISP Variable R-Bound Math for Lures
                # Y = Total Evidence
                mu_Y = crossing_val
                s2_Y = (2 * d_total * t_post) + EPS

                # X = Difference (Total Evidence - Variable Boundary)
                s2_R_bound = (sigma_r_bound**2) + EPS

                mu_X = mu_Y - r_bound_mean
                s2_X = s2_Y + s2_R_bound

                cov_XY = s2_Y

            # Build the unified 2D distribution [X, Y]
            mu_mvn = pl.array([mu_X, mu_Y])
            cov_mvn = pl.array([[s2_X, cov_XY], [cov_XY, s2_Y]])

            mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=cov_mvn, allow_singular=True)

            for j in range(1, len(clims)):
                c_upper = clims[j - 1]
                c_lower = clims[j]

                # Remember: X > 0, Confidence in Y-bin
                RLL = pl.array([0.0, c_lower])
                RUL = pl.array([INF_PROXY, c_upper])

                # Know: X < 0, Confidence in Y-bin
                KLL = pl.array([-INF_PROXY, c_lower])
                KUL = pl.array([0.0, c_upper])

                p_rem_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)
                p_know_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)

        return p_rem_conf, p_know_conf, p_new, t


param_bounds = (
    DISPAltered.Params(
        c=0.0,
        mu_r_t=0.0,          # Enforces positive evidence only for targets
        mu_f_t=0.0,
        d_r_t=EPS,
        d_f_t=EPS,
        d_l=EPS,
        tc_bound=0.0,
        r_bound_offset=0.0,
        sigma_r_bound=EPS,
        z0_t=-2.0,
        z0_l=-2.0,
        t_post=EPS,
        t0=0.0,
    ),
    DISPAltered.Params(
        c=3.0,
        mu_r_t=4.0,
        mu_f_t=4.0,
        d_r_t=2.0,
        d_f_t=2.0,
        d_l=2.0,
        tc_bound=2.0,
        r_bound_offset=3.0,
        sigma_r_bound=5.0,
        z0_t=2.0,
        z0_l=2.0,
        t_post=3.0,
        t0=1.0,
    ),
)
