# configure for compatibility with Python 3
from __future__ import absolute_import, division, print_function

# standard library imports
import shelve
from typing import ClassVar

# scientific library imports
import pylab as pl
from scipy.optimize import Bounds, differential_evolution, minimize
import pyfftw

# plotting imports
import matplotlib.pyplot as plt

# local imports
from ..config import CAA_CFG, EPS
from ..utils.multinomial_funcs import multinom_loglike, chi_square_gof
from .results import ModelEvaluation

pyfftw.interfaces.cache.enable()

data_path = "data/"
# Read in new Vincentized RT data
db = shelve.open(data_path + "neha_data.dat", "r")
DATA = db["empirical_results"]
db.close()


class BaseDDM:
    Params: ClassVar[type]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "Params" not in cls.__dict__:
            raise NotImplementedError(f"Subclass '{cls.__name__}' must define a nested 'Params' NamedTuple.")

    def __init__(self, config=CAA_CFG, use_fftw=True, nr_threads=1):
        """Initialize model with its hyperparameters."""
        self.config = config
        self.use_fftw = use_fftw
        self.nr_threads = nr_threads

    @property
    def _fft(self):
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

    @property
    def n_params(self) -> int:
        """Dynamically gets parameter count k from the subclass Params namedtuple."""
        return len(self.Params._fields)

    def evaluate(self, params, data=DATA, method="qmle", use_chisq=False) -> ModelEvaluation:
        """Computes NLL, AIC, and BIC and returns a structured evaluation dataclass."""
        neg_log_lik = self.compute_gof_all(params, data=data, method=method, use_chisq=use_chisq)
        n_params = len(params)
        n_trials = (
            len(DATA.rem_hit.rt)
            + len(DATA.know_hit.rt)
            + len(DATA.CR.rt)
            + len(DATA.miss.rt)
            + len(DATA.rem_fa.rt)
            + len(DATA.know_fa.rt)
        )

        aic = 2 * n_params + 2 * neg_log_lik
        bic = n_params * pl.log(n_trials) + 2 * neg_log_lik

        return ModelEvaluation(
            method=method,
            n_trials=n_trials,
            n_params=n_params,
            nll=neg_log_lik,
            aic=aic,
            bic=bic,
        )

    def fit(self, param_bounds, data=DATA, nr_workers=1, method="qmle", use_chisq=False):
        global_fit = self.fit_global(
            param_bounds=param_bounds, data=data, nr_workers=nr_workers, method=method, use_chisq=use_chisq
        )
        local_fit = self.fit_local(param_est=global_fit.x, data=data, method=method, use_chisq=use_chisq)
        return self.Params(*local_fit.x)

    def fit_global(self, param_bounds, data=DATA, nr_workers=1, method="qmle", use_chisq=False):
        """
        Does a global maximum-likelihood parameter search, constrained by the bounds
        listed in param_bounds, and returns the result. Each RT distribution (i.e.,
        for each judgment category and confidence level) is represented using the
        number of quantiles specified by the 'quantiles' parameter.
        """
        bounds = Bounds(*param_bounds)
        return differential_evolution(
            self.compute_gof_all,
            bounds,
            args=(data, method, use_chisq),
            workers=nr_workers,
            updating="deferred",
            strategy="best1bin",
            disp=True,
            polish=False,
        )

    def fit_local(self, param_est, data=DATA, method="qmle", use_chisq=False):
        """
        Computes MLE of params using a local (fast) and unconstrained optimization
        algorithm. Each RT distribution (i.e., for each judgment category and
        confidence level) is represented using the number of quantiles specified by
        the 'quantiles' parameter.
        """
        return minimize(
            self.compute_gof_all,
            param_est,
            args=(data, method, use_chisq),
            method="Nelder-Mead",
            options={"disp": True},
        )

    @staticmethod
    def split_params(model_params):
        """Forces the specific model to define how its unique parameters are split."""
        raise NotImplementedError

    def compute_gof_all(self, model_params, data=DATA, method="qmle", use_chisq=False):
        """
        Computes the overall goodness-of-fit of the model defined by model_params.
        This is the sum of the NLL or chi-square statistics for the distribution
        of responses to both the old and new words.
        """
        target_params, lure_params = self.split_params(model_params)

        old_data = [data.rem_hit.rt, data.know_hit.rt, data.miss.rt, data.rem_hit.conf, data.know_hit.conf]
        new_data = [data.rem_fa.rt, data.know_fa.rt, data.CR.rt, data.rem_fa.conf, data.know_fa.conf]

        gof_target = self.compute_model_gof(target_params, *old_data, method=method, use_chisq=use_chisq)
        gof_lure = self.compute_model_gof(lure_params, *new_data, method=method, use_chisq=use_chisq)

        return gof_target + gof_lure

    def compute_model_gof(
        self, model_params, rem_RTs, know_RTs, new_RTs, rem_conf, know_conf, method="qmle", use_chisq=False
    ):
        gof_func_map = {
            "qmle": lambda: self._compute_model_gof_qmle(
                model_params, rem_RTs, know_RTs, new_RTs, rem_conf, know_conf, use_chisq
            ),
            "qmpe": lambda: self._compute_model_gof_qmpe(
                model_params, rem_RTs, know_RTs, new_RTs, rem_conf, know_conf, use_chisq
            ),
            "cml": lambda: self._compute_model_gof_cml(
                model_params, rem_RTs, know_RTs, new_RTs, rem_conf, know_conf
            ),
        }
        if method not in gof_func_map:
            valid = ", ".join(f"'{k}'" for k in gof_func_map.keys())
            raise ValueError(f"Unknown fitting method: '{method}'. Valid options are: {valid}")

        return gof_func_map[method]()

    def _compute_model_gof_qmle(
        self,
        model_params,
        rem_RTs,
        know_RTs,
        new_RTs,
        rem_conf,
        know_conf,
        use_chisq=False,
    ):
        nr_quantiles = self.config.nr_quantiles

        # Total number of trials N
        N = len(rem_RTs) + len(know_RTs) + len(new_RTs)

        # Predicted quantiles and total mass for each category
        rem_qs, know_qs, new_qs, p_r, p_k, p_n = self.compute_model_quantiles(model_params)

        # Number of confidence levels being used in the model
        nr_conf_levels = len(rem_qs)

        # Pre-allocate frequency containers
        rem_freqs = pl.zeros((nr_conf_levels, nr_quantiles))
        know_freqs = pl.zeros((nr_conf_levels, nr_quantiles))

        for i in range(nr_conf_levels):
            # Isolate empirical RT vectors for the current confidence level
            r_rts = rem_RTs[rem_conf == i]
            k_rts = know_RTs[know_conf == i]

            # Append infinity to the boundaries to ensure the final quantile catches the long RT tail
            r_bins = pl.append(rem_qs[i], pl.inf)
            k_bins = pl.append(know_qs[i], pl.inf)

            # Computer number of RTs falling into each quantile bin
            rem_freqs[i], _ = pl.histogram(r_rts, bins=r_bins)
            know_freqs[i], _ = pl.histogram(k_rts, bins=k_bins)

        # Vectorized frequency count for New responses (Correct Rejections / False Alarms)
        new_bins = pl.append(new_qs, pl.inf)
        new_freqs, _ = pl.histogram(new_RTs, bins=new_bins)

        # Flip these frequencies so they are in order of descending confidence levels
        rem_freqs = pl.flipud(rem_freqs)
        know_freqs = pl.flipud(know_freqs)

        x = pl.hstack([rem_freqs.flatten(), know_freqs.flatten(), new_freqs])

        # Compute probabilities for each category (1/nr_quantiles of mass is in each bin)
        p_rem_pred = pl.repeat(p_r[:, None] / nr_quantiles, nr_quantiles, axis=1)
        p_know_pred = pl.repeat(p_k[:, None] / nr_quantiles, nr_quantiles, axis=1)
        p_new_pred = pl.repeat(p_n / nr_quantiles, nr_quantiles)

        p_pred = pl.hstack([p_rem_pred.flatten(), p_know_pred.flatten(), p_new_pred])

        if use_chisq:
            return chi_square_gof(x, N, p_pred)
        else:
            return -multinom_loglike(x, N, p_pred)

    def _compute_model_gof_qmpe(
        self,
        model_params,
        rem_RTs,
        know_RTs,
        new_RTs,
        rem_conf,
        know_conf,
        use_chisq=False,
    ):
        nr_quantiles = self.config.nr_quantiles
        N = len(rem_RTs) + len(know_RTs) + len(new_RTs)

        # Get continuous CDFs directly
        P_rem, P_know, P_new, p_rem_total, p_know_total, p_new_total, t = self.compute_model_cdfs(model_params)
        nr_conf_levels = len(P_rem)

        x_empirical = []
        p_predicted = []
        q_edges = pl.linspace(0, 1, nr_quantiles + 1)

        for i in range(nr_conf_levels):
            # We must reverse the empirical index here so Model High Conf matches Empirical High Conf.
            emp_i = (nr_conf_levels - 1) - i

            r_rts = rem_RTs[rem_conf == emp_i]
            k_rts = know_RTs[know_conf == emp_i]

            # Remember Judgments
            if len(r_rts) > 0:
                r_emp_bounds = pl.quantile(r_rts, q_edges)
                r_emp_bounds[0] = 0.0
                r_emp_bounds[-1] = pl.inf

                r_counts, _ = pl.histogram(r_rts, bins=r_emp_bounds)
                x_empirical.extend(r_counts)

                model_cdf_at_bounds = pl.append(
                    pl.interp(r_emp_bounds[:-1], t, P_rem[i], left=0.0, right=1.0), 1.0
                )
                p_predicted.extend(pl.diff(model_cdf_at_bounds) * p_rem_total[i])
            else:
                x_empirical.extend([0] * nr_quantiles)
                p_predicted.extend([p_rem_total[i] / nr_quantiles] * nr_quantiles)

            # Know Judgments
            if len(k_rts) > 0:
                k_emp_bounds = pl.quantile(k_rts, q_edges)
                k_emp_bounds[0] = 0.0
                k_emp_bounds[-1] = pl.inf

                k_counts, _ = pl.histogram(k_rts, bins=k_emp_bounds)
                x_empirical.extend(k_counts)

                model_cdf_at_bounds = pl.append(
                    pl.interp(k_emp_bounds[:-1], t, P_know[i], left=0.0, right=1.0), 1.0
                )
                p_predicted.extend(pl.diff(model_cdf_at_bounds) * p_know_total[i])
            else:
                x_empirical.extend([0] * nr_quantiles)
                p_predicted.extend([p_know_total[i] / nr_quantiles] * nr_quantiles)

        # New Responses (Lures) do not have confidence splits in this block, so no reverse index needed
        if len(new_RTs) > 0:
            new_emp_bounds = pl.quantile(new_RTs, q_edges)
            new_emp_bounds[0] = 0.0
            new_emp_bounds[-1] = pl.inf

            new_counts, _ = pl.histogram(new_RTs, bins=new_emp_bounds)
            x_empirical.extend(new_counts)

            model_cdf_at_bounds = pl.append(pl.interp(new_emp_bounds[:-1], t, P_new, left=0.0, right=1.0), 1.0)
            p_predicted.extend(pl.diff(model_cdf_at_bounds) * p_new_total)
        else:
            x_empirical.extend([0] * nr_quantiles)
            p_predicted.extend([p_new_total / nr_quantiles] * nr_quantiles)

        x = pl.array(x_empirical)
        p = pl.clip(pl.array(p_predicted), EPS, 1.0)

        if use_chisq:
            return chi_square_gof(x, N, p)
        else:
            return -multinom_loglike(x, N, p)

    def _compute_model_gof_cml(
        self,
        model_params,
        rem_RTs,
        know_RTs,
        new_RTs,
        rem_conf,
        know_conf,
        lapse_rate=0.02,  # 2% uniform mixture to prevent log(0) crashes
    ):
        # Generate the probability mass per time step
        p_rem_conf, p_know_conf, p_new, t = self.predicted_proportions(model_params)

        delta_t = self.config.delta_t
        max_t = self.config.max_t
        nr_conf_levels = len(p_rem_conf)

        # Total response categories: Remember(3) + Know(3) + New(1) = 7
        num_choices = (2 * nr_conf_levels) + 1

        # Convert probability mass to continuous probability density
        pdf_rem = p_rem_conf / delta_t
        pdf_know = p_know_conf / delta_t
        pdf_new = p_new / delta_t

        # The penalty density for outliers/random guessing
        lapse_density = lapse_rate * (1.0 / num_choices) * (1.0 / max_t)

        total_nll = 0.0

        for i in range(nr_conf_levels):
            # Reverse index to match your empirical coding
            emp_i = (nr_conf_levels - 1) - i

            # Remember Responses
            r_rts = rem_RTs[rem_conf == emp_i]
            if len(r_rts) > 0:
                # Interpolate exact RTs (left=0.0, right=0.0 catches max_t violations)
                r_vals = pl.interp(r_rts, t, pdf_rem[i], left=0.0, right=0.0)
                r_mixed = (1.0 - lapse_rate) * r_vals + lapse_density
                total_nll -= pl.sum(pl.log(r_mixed))

            # Know Responses
            k_rts = know_RTs[know_conf == emp_i]
            if len(k_rts) > 0:
                k_vals = pl.interp(k_rts, t, pdf_know[i], left=0.0, right=0.0)
                k_mixed = (1.0 - lapse_rate) * k_vals + lapse_density
                total_nll -= pl.sum(pl.log(k_mixed))

        # New Responses
        if len(new_RTs) > 0:
            n_vals = pl.interp(new_RTs, t, pdf_new, left=0.0, right=0.0)
            n_mixed = (1.0 - lapse_rate) * n_vals + lapse_density
            total_nll -= pl.sum(pl.log(n_mixed))

        return total_nll

    def compute_model_cdfs(self, model_params):
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

        return P_rem, P_know, P_new, p_rem_total, p_know_total, p_new_total, t

    def compute_model_quantiles(self, model_params):
        """Run the model and extract RT boundaries for each category."""
        P_rem, P_know, P_new, p_rem_total, p_know_total, p_new_total, t = self.compute_model_cdfs(model_params)

        quantile_inc = 1.0 / self.config.nr_quantiles
        quantiles = pl.arange(0, 1, quantile_inc)

        # Find time point for each quantile
        # Cumsum is already sorted, so we can use searchsorted
        max_idx = len(t) - 1
        rem_quantiles = pl.array(
            [t[pl.clip(pl.searchsorted(P_rem[i], quantiles), 0, max_idx)] for i in range(len(P_rem))]
        )
        know_quantiles = pl.array(
            [t[pl.clip(pl.searchsorted(P_know[i], quantiles), 0, max_idx)] for i in range(len(P_know))]
        )
        new_quantiles = t[pl.clip(pl.searchsorted(P_new, quantiles), 0, max_idx)]

        # Initialize first quantile at t=0
        rem_quantiles[:, 0] = 0
        know_quantiles[:, 0] = 0
        new_quantiles[0] = 0

        return rem_quantiles, know_quantiles, new_quantiles, p_rem_total, p_know_total, p_new_total

    def predicted_proportions(self, params):
        """Generates the predicted probability distributions."""
        raise NotImplementedError

    def plot_rt_distributions(self, params, show=True):
        """Plots predicted RT probability density distributions for target and lure stimuli."""
        delta_t = self.config.delta_t

        # Calculate predictions
        params_target, params_lure = self.split_params(params)
        p_rem, p_know, p_new, t = self.predicted_proportions(params_target)
        p_rem_lures, p_know_lures, p_new_lures, t = self.predicted_proportions(params_lure)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Dynamically handle confidence levels
        nr_conf_levels = len(p_rem)
        confidence_labels = [f"conf={nr_conf_levels - 1 - i}" for i in range(nr_conf_levels)]

        # Generate smooth color gradients matching confidence levels
        colors_rem = plt.colormaps["Reds"](pl.linspace(1, 0.4, nr_conf_levels))
        colors_know = plt.colormaps["Blues"](pl.linspace(1, 0.4, nr_conf_levels))

        # Data mapping to eliminate duplicated code
        datasets = [
            (axes[0], p_rem, p_know, p_new, "RT Distributions for Target Words"),
            (axes[1], p_rem_lures, p_know_lures, p_new_lures, "RT Distributions for Lure Words"),
        ]

        for ax, pr, pk, pn, title in datasets:
            for i in range(nr_conf_levels):
                ax.plot(t, pr[i] / delta_t, color=colors_rem[i], label=f"rem {confidence_labels[i]}")
                ax.plot(t, pk[i] / delta_t, color=colors_know[i], label=f"know {confidence_labels[i]}")

            ax.plot(t, pn / delta_t, color="black", label="new")
            ax.set_title(title)
            ax.set_xlabel("Reaction Time (sec)")
            ax.set_ylabel("p(RT judgment)")
            ax.legend(fontsize=7)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, axes
