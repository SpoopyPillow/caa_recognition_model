from caa_model import dual_process as dp
from caa_model.dual_process import compute_model_gof
from caa_model.dual_process import predicted_proportions_sim as pps
from caa_model.dual_process import predicted_proportions as pp
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd

from caa_model.dual_process import INF_PROXY, EPS, DELTA_T, MAX_T, NR_TSTEPS, NR_SSTEPS
from caa_model import fftw_test as fftw
import pylab as pl
from scipy import stats

NR_BINS = 5  # 5 confidence bins as discussed

# Same calling method as before, you would have something like
# params_old = [c, mu_r, d_t, tc_bound, r_bound_offset,  z0, deltaT, t0]
# params_new = [c, mu_r_lure, d_l, tc_bound, r_bound_offset, -z0, deltaT, t0]

# predicted_proportions_revised(*params_old)
# predicted_proportions_revised(*params_new)

def get_rect_prob(rv, low, high):
    """
    Get probability of a random variable in a rectangle
    """
    pts = pl.array([
        [high[0], high[1]],
        [low[0],  high[1]],
        [high[0], low[1]],
        [low[0],  low[1]]
    ])
    p = rv.cdf(pts)
    return p[0] - p[1] - p[2] + p[3]


def predicted_proportions_revised(c, mu_r, d, tc_bound, r_bound_offset, z0, deltaT, sigma_z0, t0=0):
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

    #creating bin edges
    c = pl.array(c, ndmin=1)
    n = len(c)
    clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))

    #remember criterion position
    r_bound = z0 + r_bound_offset

    #standard deviation and mean drift of the accumulator per-step
    sigma = pl.sqrt(2 * d * DELTA_T) 
    mu    = mu_r * DELTA_T

    #making the time axis, to_idx is the index of the first time step
    t      = pl.linspace(DELTA_T, MAX_T, NR_TSTEPS)
    to_idx = pl.argmin((t - t0) ** 2)
    bound  = pl.exp(-tc_bound * pl.clip(t - t0, 0, None))
    #bound is the collapsing boundary at each time point

    #making the grid for the accumulator
    space_lim = max(bound) + 3 * sigma
    delta_s   = 2 * space_lim / NR_SSTEPS
    x         = pl.linspace(-space_lim, space_lim, NR_SSTEPS)
    
    #diffusion kernel pre-fft'd
    kernel    = stats.norm.pdf(x, mu, sigma) * delta_s
    ft_kernel = fftw.fft(kernel)

    #initializing output arrays
  
    tx          = pl.zeros((len(t), len(x)))
    p_old       = pl.zeros(pl.shape(t))
    p_new       = pl.zeros(pl.shape(t))
    p_rem_conf  = pl.zeros((n + 1, pl.size(t)))
    p_know_conf = pl.zeros((n + 1, pl.size(t)))

    ############################################
    ## take care of the first timestep        ##
    ############################################

    #seeds the particle distribution at t0 and convernts the probability density into probability mass per grid cell
    #the sigma_init here widens the distributio we pull from, since different trials can start at differfent starting points drawn from N(z0, sigma_z0)
    sigma_init = pl.sqrt(sigma**2 + sigma_z0**2)
    tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma_init) * delta_s
    
    #any particles that crossed a boundary already (on the first step) are recorded as immediate old/new responses
    p_old[to_idx] = pl.sum(tx[to_idx][x >=  bound[to_idx]])
    p_new[to_idx] = pl.sum(tx[to_idx][x <= -bound[to_idx]])

    #expected accumulator positions given the particle just crossed bound[to_idx], here rh=1 and mu_f=0 so this equation reduces
    mu_r_cond = mu_r * t[to_idx] + (bound[to_idx] - t[to_idx] * mu_r - z0)

    #zeroes out particles that already crossed so they don't contribute to future steps
    tx[to_idx] *= (abs(x) < bound[to_idx])

    #Expected position of accumulator after deltaT more seconds (for the recollection dimension and the combined dimension)
    mu_r_delta    = mu_r_cond + mu_r * deltaT
    mu_comb_delta = mu_r * deltaT + bound[to_idx]

    #Variance of the accumulator after deltaT 
    s2_r_delta    = EPS**2 + 2 * d * deltaT
    cov_delta     = s2_r_delta
    s2_comb_delta = s2_r_delta + s2_r_delta
    mu_mvn    = pl.array([mu_r_delta, mu_comb_delta])
    sigma_mvn = pl.array([[s2_r_delta, cov_delta], [cov_delta, s2_comb_delta]])
    mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=sigma_mvn, allow_singular=True)

    #For each confidence band, integrates the bivariate normal over the know rectangle and the remember rectangle, 
    #multiplied by p_old[to_idx] to get joint probability of crossing and landing in the category
    for j in range(1, len(clims)):
        KLL = pl.array([-INF_PROXY, clims[j]])
        KUL = pl.array([r_bound,    clims[j - 1]])
        RLL = pl.array([r_bound,    clims[j]])
        RUL = pl.array([INF_PROXY,  clims[j - 1]])
        p_know_conf[j-1, to_idx] = p_old[to_idx] * get_rect_prob(mvn_dist, KLL, KUL)
        p_rem_conf [j-1, to_idx] = p_old[to_idx] * get_rect_prob(mvn_dist, RLL, RUL)

    #######################################
    ## take care of subsequent timesteps ##
    #######################################
    for i in range(to_idx + 1, len(t)):
        #convolves previous step's particle distribution with the diffusion kernel
        tx[i] = abs(pl.ifftshift(fftw.ifft(fftw.fft(tx[i-1]) * ft_kernel)))

        #Extracts the particles that crossed the upper boundary on this step, comb_est is the the expectation of a particle
        #given that it crossed the upper boundary
        p_pos    = tx[i][x >=  bound[i]]
        x_pos    = x[x >= bound[i]]
        comb_est = (pl.dot(p_pos, x_pos) + EPS) / (pl.sum(p_pos) + EPS)

        #Total probability mass crossing the upper boundary for old and new responses
        p_old[i] = pl.sum(p_pos)
        p_new[i] = pl.sum(tx[i][x <= -bound[i]])

        #Expected accumulator position given crossing at comb_est
        mu_r_cond = mu_r * t[i] + (comb_est - t[i] * mu_r - z0)

        #Zeroes out particles that already crossed the boundary
        tx[i] *= (abs(x) < bound[i])

        #Expected position of accumulator after deltaT more seconds (for the recollection dimension and the combined dimension)
        mu_r_delta    = mu_r_cond + mu_r * deltaT
        mu_comb_delta = mu_r * deltaT + comb_est
        s2_r_delta    = 2 * d * deltaT
        cov_delta     = s2_r_delta
        s2_comb_delta = s2_r_delta + s2_r_delta
        mu_mvn    = pl.array([mu_r_delta, mu_comb_delta])
        sigma_mvn = pl.array([[s2_r_delta, cov_delta], [cov_delta, s2_comb_delta]])
        mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=sigma_mvn, allow_singular=True)

        #same thing as for the loop in the first time step
        for j in range(1, len(clims)):
            KLL = pl.array([-INF_PROXY, clims[j]])
            KUL = pl.array([r_bound,    clims[j - 1]])
            RLL = pl.array([r_bound,    clims[j]])
            RUL = pl.array([INF_PROXY,  clims[j - 1]])
            p_know_conf[j-1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)
            p_rem_conf [j-1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)


    #sums over the confidence dimension, giving the marginal RT distribution for remember and know responses regardless of confidence level
    #these were computed but unused in the original old model implimentation, so I'm leaving it here

    # p_remember = p_rem_conf.sum(0)
    # p_know     = p_know_conf.sum(0)

    return p_rem_conf, p_know_conf, p_new, t
