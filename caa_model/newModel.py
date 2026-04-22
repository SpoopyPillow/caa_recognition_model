import shelve
from collections import namedtuple
import pylab as pl
from scipy import stats, optimize
from . import fftw_test as fftw
from .multinomial_funcs import multinom_loglike, chi_square_gof, get_rect_prob

data_path = "caa_model/data/"

db = shelve.open(data_path + "neha_data.dat", "r")
DATA = db["empirical_results"]
db.close()

NR_BINS = 5  # 5 confidence bins as discussed
INF_PROXY = 10  # a value used to provide very large but finite bounds for mvn integration
EPS = 1e-10  # a very small value (used for numerical stability)
NR_THREADS = 1  # this is for multithreaded fft
DELTA_T = 0.025  # size of discrete time increment (sec.)
MAX_T = 8.0  # ceil(percentile(all_RT,99.5))
NR_TSTEPS = int(MAX_T / DELTA_T)
NR_SSTEPS = 8192
NR_SAMPLES = 10000  # number of trials to use for MC likelihood computation
NR_WORKERS = 8  # Number of cores for optimizing parameters

NR_QUANTILES = 10

fftw.fftw_setup(pl.zeros(NR_SSTEPS), NR_THREADS)

CAAParams = namedtuple(
    "CAAParams",
    [
        "c",  # High confidence boundary
        "mu_t",  # Target drift
        "mu_l",  # Lure drift
        "d",  # Diffusion constant
        "tc_bound",  # Boundary collapse rate
        "r_bound_offset",  # Remember boundary offset
        "z0_t",  # Target starting point
        "z0_l",  # Lure starting point
        "deltaT",  # Post-decision accumulation time
        "sigma_z0",  # Starting position variability
        "t0",  # Non-decision time
    ],
)

param_bounds = CAAParams(
    c=(0.0, 2.0),
    mu_t=(0.0, 2.0),
    mu_l=(0.0, 1.0),
    d=(EPS, 1.0),
    tc_bound=(0.0, 1.0),
    r_bound_offset=(0.0, 1.0),
    z0_t=(-2.0, 2.0),
    z0_l=(-2.0, 2.0),
    deltaT=(EPS, 2.0),
    sigma_z0=(EPS, 1.0),
    t0=(0.0, 0.5),
)

# Parameters obtained from using 10 quantiles
params_est = CAAParams(
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


def find_ml_params_all(quantiles=NR_QUANTILES, use_chisq=True):
    """
    Does a global maximum-likelihood parameter search, constrained by the bounds
    listed in param_bounds, and returns the result. Each RT distribution (i.e.,
    for each judgment category and confidence level) is represented using the
    number of quantiles specified by the 'quantiles' parameter.
    """
    return optimize.differential_evolution(
        compute_gof_all,
        param_bounds,
        args=(quantiles, DATA, use_chisq),
        workers=NR_WORKERS,
        updating="deferred",
        strategy="best1bin",
        disp=True,
    )


def find_ml_params_all_lm(quantiles=NR_QUANTILES, use_chisq=True):
    """
    Computes MLE of params using a local (fast) and unconstrained optimization
    algorithm. Each RT distribution (i.e., for each judgment category and
    confidence level) is represented using the number of quantiles specified by
    the 'quantiles' parameter.
    """
    return optimize.fmin(compute_gof_all, params_est, args=(quantiles, DATA, use_chisq), disp=True)


def compute_gof_all(model_params, quantiles=NR_QUANTILES, data=DATA, use_chisq=True):
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

    gof_target = compute_model_gof(target_params, *old_data, nr_quantiles=quantiles, use_chisq=use_chisq)
    gof_lure = compute_model_gof(lure_params, *new_data, nr_quantiles=quantiles, use_chisq=use_chisq)

    return gof_target + gof_lure


def compute_model_gof(
    model_params, rem_RTs, know_RTs, new_RTs, rem_conf, know_conf, nr_quantiles=NR_QUANTILES, use_chisq=True
):
    """
    Compute the NLL or chi-square fit of the model to the data.
    """

    # Total number of trials N
    N = len(rem_RTs) + len(know_RTs) + len(new_RTs)

    # Predicted quantiles and total mass for each category
    rem_qs, know_qs, new_qs, p_r, p_k, p_n = compute_model_quantiles(model_params, nr_quantiles)

    # Number of confidence levels being used in the model
    nr_conf_levels = len(rem_qs)

    # Adjust the number of confidence levels in the data to match
    rem_conf = pl.clip(rem_conf, 0, nr_conf_levels - 1)
    know_conf = pl.clip(know_conf, 0, nr_conf_levels - 1)

    # Computer number of RTs falling into each quantile bin
    rem_freqs = pl.array(
        [-pl.diff([pl.sum(rem_RTs[rem_conf == i] > q) for q in rem_qs[i]] + [0]) for i in range(nr_conf_levels)]
    )
    know_freqs = pl.array(
        [-pl.diff([pl.sum(know_RTs[know_conf == i] > q) for q in know_qs[i]] + [0]) for i in range(nr_conf_levels)]
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


def compute_model_quantiles(model_params, nr_quantiles=NR_QUANTILES):
    """
    Run the model and extract RT boundaries for each category.
    """

    quantile_inc = 1.0 / nr_quantiles
    quantiles = pl.arange(0, 1, quantile_inc)

    # Call the model
    p_rem_conf, p_know_conf, p_new, t = predicted_proportions(*model_params)

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


def predicted_proportions(c, mu_r, d, tc_bound, r_bound_offset, z0, deltaT, sigma_z0, t0=0):
    """
    Revised single-accumulator DISP CAA model (Glass, 2026).

    Call twice (once for targets, once for lures) with the appropriate
    parameter values, exactly as the original predicted_proportions() is used.
    For lures, pass mu_r=mu_r_lure, d=d_l, z0=-z0.

    Parameters:
    c              : confidence boundary (C_High); scalar or 1-D array
    mu_r           : mean recollection drift rate
    d              : diffusion constant
    tc_bound       : boundary collapse rate (tau)
    r_bound_offset : fixed distance from z0 to remember criterion
    z0             : starting location (recency > 0, novelty < 0)
    deltaT         : post-response accumulation interval
    t0             : accumulation start time
    sigma_z0       : variability in percieved familiarity across trials
    """

    # Create confidence bins
    c = pl.array(c, ndmin=1)
    n = len(c)
    clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

    # Set remember criterion position
    r_bound = z0 + r_bound_offset

    # Standard deviation and mean drift of the accumulator per-step
    sigma = pl.sqrt(2 * d * DELTA_T)
    mu = mu_r * DELTA_T

    # Create the time axis, where to_idx is the index of the first time step
    t = pl.linspace(DELTA_T, MAX_T, NR_TSTEPS)
    to_idx = pl.argmin((t - t0) ** 2)
    # Bound is the collapsing boundary at each time point
    bound = pl.exp(-tc_bound * pl.clip(t - t0, 0, None))

    # Create the grid for the accumulator
    space_lim = max(bound) + 3 * sigma
    delta_s = 2 * space_lim / NR_SSTEPS
    x = pl.linspace(-space_lim, space_lim, NR_SSTEPS)

    # Kernel is the probability mass function for each step
    kernel = stats.norm.pdf(x, mu, sigma) * delta_s
    # FFT to prepare for convolution
    ft_kernel = fftw.fft(kernel)

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
            tx[i] = abs(pl.ifftshift(fftw.ifft(fftw.fft(tx[i - 1]) * ft_kernel)))

        # Extract particles that crossed boundaries
        p_pos = tx[i][x >= bound[i]]
        p_old[i] = pl.sum(p_pos)
        p_new[i] = pl.sum(tx[i][x <= -bound[i]])

        # Different calculation on first step
        if i == to_idx:
            # Crossing is exactly on the boundary
            crossing_val = bound[to_idx]
        else:
            # Calculate expected crossing position
            x_pos = x[x >= bound[i]]
            crossing_val = (pl.dot(p_pos, x_pos) + EPS) / (pl.sum(p_pos) + EPS)

        # Zero out particles that already crossed the boundary
        tx[i] *= abs(x) < bound[i]

        # Mean and SD of accumulator after deltaT
        mu_delta = crossing_val + mu_r * deltaT
        std_delta = pl.sqrt(2 * d * deltaT)

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
