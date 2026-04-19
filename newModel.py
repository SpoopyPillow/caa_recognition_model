import pylab as pl
from scipy import stats
from . import fftw_test as fftw
from .multinomial_funcs import multinom_loglike, chi_square_gof, get_rect_prob

NR_BINS = 5  # 5 confidence bins as discussed
INF_PROXY = 10  # a value used to provide very large but finite bounds for mvn integration
EPS = 1e-10  # a very small value (used for numerical stability)
NR_THREADS = 1  # this is for multithreaded fft
DELTA_T = 0.025  # size of discrete time increment (sec.)
MAX_T = 8.0  # ceil(percentile(all_RT,99.5))
NR_TSTEPS = int(MAX_T / DELTA_T)
NR_SSTEPS = 8192
NR_SAMPLES = 10000  # number of trials to use for MC likelihood computation

NR_QUANTILES = 10


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
    p_know_conf = pl.zeros((n + 1, pl.size(t))) # Yes responses that are known (does not cross r_bound)

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
