# configure for compatibility with Python 3
from __future__ import absolute_import, division, print_function

# standard library imports
from collections import namedtuple

# scientific library imports
import pylab as pl
import numpy
from scipy import stats
from scipy import optimize
import pyfftw

# local imports
from ..config import CAA_CFG, EPS, INF_PROXY
from ..utils import fftw_test as fftw
from ..utils.multinomial_funcs import multinom_loglike, chi_square_gof, get_rect_prob

pyfftw.interfaces.cache.enable()


class BaseDDM:
    def __init__(self, config=CAA_CFG, use_fftw=True, nr_threads=1):
        """Initialize model with its hyperparameters."""
        self.config = config
        self.use_fftw = use_fftw
        self.nr_threads = nr_threads
        self.best_params = None

    @property
    def _fftw(self):
        if self.use_fftw:
            return lambda x: pyfftw.interfaces.numpy_fft.fft(x, threads=self.nr_threads)
        return pl.fft

    @property
    def _ifft(self):
        if self.use_fftw:
            return lambda x: pyfftw.interfaces.numpy_fft.ifft(x, threads=self.nr_threads).real
        return lambda x: pl.ifft(x).real

    @property
    def _fft2(self):
        if self.use_fftw:
            return lambda x: pyfftw.interfaces.numpy_fft.fft2(x, threads=self.nr_threads)
        return pl.fft2

    @property
    def _ifft2(self):
        if self.use_fftw:
            return lambda x: pyfftw.interfaces.numpy_fft.ifft2(x, threads=self.nr_threads).real
        return lambda x: pl.ifft2(x).real

    def fit_global(self, param_bounds, data, nr_workers=1, use_chisq=True):
        """
        Does a global maximum-likelihood parameter search, constrained by the bounds
        listed in param_bounds, and returns the result. Each RT distribution (i.e.,
        for each judgment category and confidence level) is represented using the
        number of quantiles specified by the 'quantiles' parameter.
        """
        return optimize.differential_evolution(
            self.compute_gof_all,
            param_bounds,
            args=(data, use_chisq),
            workers=nr_workers,
            updating="deferred",
            strategy="best1bin",
            disp=True,
        )

    def fit_local(self, param_est, data, use_chisq=True):
        """
        Computes MLE of params using a local (fast) and unconstrained optimization
        algorithm. Each RT distribution (i.e., for each judgment category and
        confidence level) is represented using the number of quantiles specified by
        the 'quantiles' parameter.
        """
        return optimize.fmin(self.compute_gof_all, param_est, args=(data, use_chisq), disp=True)

    def compute_gof_all(self, model_params, data, use_chisq=True):
        """
        Computes the overall goodness-of-fit of the model defined by model_params.
        This is the sum of the NLL or chi-square statistics for the distribution
        of responses to both the old and new words.
        """
        c_val, mu_t, mu_l, d, tc, r_off, z0_t, z0_l, dT, s_z0, t0 = model_params
        # Confidence bins (High, Med, Low)
        c_list = [c_val, 0]

        target_params = (c_list, mu_t, d, tc, r_off, z0_t, dT, s_z0, t0)
        old_data = [data.rem_hit.rt, data.know_hit.rt, data.miss.rt, data.rem_hit.conf, data.know_hit.conf]

        lure_params = (c_list, mu_l, d, tc, r_off, z0_l, dT, s_z0, t0)
        new_data = [data.rem_fa.rt, data.know_fa.rt, data.CR.rt, data.rem_fa.conf, data.know_fa.conf]

        gof_target = self.compute_model_gof(target_params, *old_data, use_chisq=use_chisq)
        gof_lure = self.compute_model_gof(lure_params, *new_data, use_chisq=use_chisq)

        return gof_target + gof_lure

    def compute_model_gof(
        self,
        model_params,
        rem_RTs,
        know_RTs,
        new_RTs,
        rem_conf,
        know_conf,
        use_chisq=True,
    ):
        """Compute the NLL or chi-square fit of the model to the data."""
        nr_quantiles = self.config.nr_quantiles

        # Total number of trials N
        N = len(rem_RTs) + len(know_RTs) + len(new_RTs)

        # Predicted quantiles and total mass for each category
        rem_qs, know_qs, new_qs, p_r, p_k, p_n = self.compute_model_quantiles(model_params)

        # Number of confidence levels being used in the model
        nr_conf_levels = len(rem_qs)

        # Adjust the number of confidence levels in the data to match
        rem_conf = pl.clip(rem_conf, 0, nr_conf_levels - 1)
        know_conf = pl.clip(know_conf, 0, nr_conf_levels - 1)

        # Computer number of RTs falling into each quantile bin
        rem_freqs = pl.array(
            [
                -pl.diff([pl.sum(rem_RTs[rem_conf == i] > q) for q in rem_qs[i]] + [0])
                for i in range(nr_conf_levels)
            ]
        )
        know_freqs = pl.array(
            [
                -pl.diff([pl.sum(know_RTs[know_conf == i] > q) for q in know_qs[i]] + [0])
                for i in range(nr_conf_levels)
            ]
        )
        new_freqs = -pl.diff([pl.sum(new_RTs > q) for q in new_qs] + [0])

        # Flip these frequencies so they are in order of descending confidence levels
        rem_freqs = pl.flipud(rem_freqs)
        know_freqs = pl.flipud(know_freqs)

        x = pl.hstack([rem_freqs.flatten(), know_freqs.flatten(), new_freqs])

        # Compute probabilities for each category (1/nr_quantiles of mass is in each bin)
        p_rem_pred = p_r[:, None] * pl.ones((nr_conf_levels, nr_quantiles)) / float(nr_quantiles)
        p_know_pred = p_k[:, None] * pl.ones((nr_conf_levels, nr_quantiles)) / float(nr_quantiles)
        p_new_pred = p_n * pl.ones(nr_quantiles) / float(nr_quantiles)

        p_pred = pl.hstack([p_rem_pred.flatten(), p_know_pred.flatten(), p_new_pred])

        if use_chisq:
            return chi_square_gof(x, N, p_pred)
        else:
            return -multinom_loglike(x, N, p_pred)

    def compute_model_quantiles(self, model_params):
        """Run the model and extract RT boundaries for each category."""

        quantile_inc = 1.0 / self.config.nr_quantiles
        quantiles = pl.arange(0, 1, quantile_inc)

        # Call the model
        p_rem_conf, p_know_conf, p_new, t = self.predicted_proportions(model_params)

        # Calculate mass per confidence bin
        p_rem_total = pl.sum(p_rem_conf, axis=-1) + EPS
        p_know_total = pl.sum(p_know_conf, axis=-1) + EPS
        p_new_total = pl.sum(p_new) + EPS

        # Create CDF
        P_rem = pl.cumsum(p_rem_conf, axis=-1) / p_rem_total[:, None]
        P_know = pl.cumsum(p_know_conf, axis=-1) / p_know_total[:, None]
        P_new = pl.cumsum(p_new) / p_new_total

        # Find time point for each quantile
        rem_quantiles = pl.array([t[pl.argmax(P_rem > q, -1)] for q in quantiles]).T
        know_quantiles = pl.array([t[pl.argmax(P_know > q, -1)] for q in quantiles]).T
        new_quantiles = pl.array([t[pl.argmax(P_new > q)] for q in quantiles])

        # Initialize first quantile at t=0
        rem_quantiles[:, 0] = 0
        know_quantiles[:, 0] = 0
        new_quantiles[0] = 0

        return rem_quantiles, know_quantiles, new_quantiles, p_rem_total, p_know_total, p_new_total

    def predicted_proportions(self, params):
        """Generates the predicted probability distributions."""
        raise NotImplementedError
