# configure for compatibility with Python 3
from __future__ import absolute_import, division, print_function

# standard library imports
from collections import namedtuple

# scientific library imports
import pylab as pl
from scipy import stats

# local imports
from .base_model import BaseDDM
from ..config import EPS, INF_PROXY
from ..utils.multinomial_funcs import get_rect_prob


class DISPClassicDDM(BaseDDM):
    Params = namedtuple(
        "DISPClassicParams",
        [
            "c",
            "mu_r",
            "mu_f",
            "d_r",
            "d_f",
            "tc_bound",
            "r_bound",
            "z0",
            "mu_r_new",
            "mu_f_new",
            "deltaT",
            "t_offset",
        ],
    )

    @staticmethod
    def split_params(model_params):
        c, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, mu_r0, mu_f0, deltaT, t_offset = model_params
        c_list = [c, 0]

        target_params = (c_list, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, deltaT, t_offset)
        lure_params = (c_list, mu_r0, mu_f0, d_r, d_f, tc_bound, r_bound, z0, deltaT, t_offset)

        return target_params, lure_params

    def predicted_proportions(self, params):
        c, mu_r, mu_f, d_r, d_f, tc_bound, r_bound, z0, deltaT, t_offset = params
        # make c (the confidence levels) an array in case it is a scalar value
        c = pl.array(c, ndmin=1)
        n = len(c)
        # form an array consisting of the appropriate (upper) integration limits
        clims = pl.hstack(([INF_PROXY], c, [-INF_PROXY]))
        # compute process SD
        sigma_r = pl.sqrt(2 * d_r * self.config.delta_t)
        sigma_f = pl.sqrt(2 * d_f * self.config.delta_t)
        sigma = pl.sqrt(sigma_r**2 + sigma_f**2)

        # compute the correlation for r given r+f
        rho = sigma_r / sigma

        t = pl.linspace(self.config.delta_t, self.config.max_t, self.config.nr_tsteps)
        # this is the time axis
        to_idx = pl.argmin((t - t_offset) ** 2)
        # compute the index for t_offset
        bound = pl.exp(-tc_bound * pl.clip(t - t_offset, 0, None))
        # this is the collapsing bound

        mu = (mu_r + mu_f) * self.config.delta_t
        # this is the average overall drift rate, with r = 'recall' and f = 'familiar'
        # compute the bounding limit of the space domain. This should include at least 99% of the probability mass when the particle is at the largest possible bound
        space_lim = max(bound) + 3 * sigma
        delta_s = 2 * space_lim / self.config.nr_ssteps
        # finally, construct the space axis
        x = pl.linspace(-space_lim, space_lim, self.config.nr_ssteps)
        # compute the diffusion kernel
        kernel = stats.norm.pdf(x, mu, sigma) * delta_s
        # ... and its Fourier transform. We'll use this to compute FD convolutions
        ft_kernel = self._fft(kernel)
        tx = pl.zeros((len(t), len(x)))

        # Construct arrays to hold RT distributions
        p_old = pl.zeros(pl.shape(t))
        p_new = pl.zeros(pl.shape(t))
        p_rem_conf = pl.zeros((n + 1, pl.size(t)))
        p_know_conf = pl.zeros((n + 1, pl.size(t)))

        ############################################
        ## take care of the first timestep #########
        ############################################
        tx[to_idx] = stats.norm.pdf(x, mu + z0, sigma) * delta_s
        p_old[to_idx] = pl.sum(tx[to_idx][x >= bound[to_idx]])
        p_new[to_idx] = pl.sum(tx[to_idx][x <= -bound[to_idx]])
        # compute STD(r) for the current time
        s_r = sigma_r
        s_f = sigma_f
        # compute STD(r+f) for the current time
        s_comb = sigma
        # compute E[r|(r+f)]
        # mu_r_cond = mu_r*t[0]+rho*s_r*(bound[0]-t[0]*(mu_r+mu_f))/s_comb;
        mu_r_cond = mu_r * t[to_idx] + (bound[to_idx] - t[to_idx] * (mu_r + mu_f) - z0) * rho**2
        # compute STD[r|(r+f)]
        s_r_cond = s_r * pl.sqrt(1 - rho**2)
        s_f_cond = s_f * pl.sqrt(1 - (sigma_f / sigma) ** 2)

        # remove from consideration any particles that already hit the bound
        tx[to_idx] *= abs(x) < bound[to_idx]

        ############################################################################
        # compute the parameters of the bivariate distribution of particle locations
        # deltaT seconds after old/new decision

        mu_r_delta = mu_r_cond + mu_r * deltaT
        mu_comb_delta = (mu_r + mu_f) * deltaT + bound[to_idx]
        s2_r_delta = s_r_cond**2 + 2 * d_r * deltaT
        s2_f_delta = s_f_cond**2 + 2 * d_f * deltaT
        s2_comb_delta = s2_r_delta + s2_f_delta
        # s2_comb_delta = 2*deltaT*(d_r+d_f);
        # s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
        cov_delta = s2_r_delta
        mu_mvn = pl.array([mu_r_delta, mu_comb_delta])
        sigma_mvn = pl.array([[s2_r_delta, cov_delta], [cov_delta, s2_comb_delta]])
        mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=sigma_mvn, allow_singular=True)
        ############################################################################
        for j in range(1, len(clims)):
            # Note that the clims appear in descending order, from highest to lowest value
            KLL = pl.array([-INF_PROXY, clims[j]])
            # lower limit for 'know' class
            KUL = pl.array([r_bound, clims[j - 1]])
            # upper limit for 'know' class
            RLL = pl.array([r_bound, clims[j]])
            # lower limit for 'remember' class
            RUL = pl.array([INF_PROXY, clims[j - 1]])
            # upper limit for 'remember' class
            p_know_conf[j - 1, to_idx] = p_old[to_idx] * get_rect_prob(mvn_dist, KLL, KUL)
            p_rem_conf[j - 1, to_idx] = p_old[to_idx] * get_rect_prob(mvn_dist, RLL, RUL)

        #######################################
        ## take care of subsequent timesteps ##
        #######################################

        for i in range(to_idx + 1, len(t)):
            # tx[i] = convolve(tx[i-1],kernel,'same');
            # convolve the particle distribution from the previous timestep
            # with the diffusion kernel (using Fourier domain convolution)
            tx[i] = abs(pl.ifftshift(self._ifft(self._fft(tx[i - 1]) * ft_kernel)))

            p_pos = tx[i][x >= bound[i]]
            # probability of each particle position above the upper bound
            x_pos = x[x >= bound[i]]
            # location of each particle position above the upper bound

            # compute the expected value of a particle that just exceeded the bound
            # during the last time interval
            comb_est = (pl.dot(p_pos, x_pos) + EPS) / (pl.sum(p_pos) + EPS)

            p_old[i] = pl.sum(p_pos)
            # total probability that particle crosses upper bound
            p_new[i] = pl.sum(tx[i][x <= -bound[i]])
            # probability that particle crosses lower bound

            # compute STD(r) for the current time
            s_r = pl.sqrt(2 * d_r * t[i])
            s_f = pl.sqrt(2 * d_f * t[i])
            # compute STD[r|(r+f)]
            s_r_cond = s_r * pl.sqrt(1 - rho**2)
            s_f_cond = s_f * pl.sqrt(1 - (sigma_f / sigma) ** 2)
            # compute E[r|(r+f)]
            mu_r_cond = mu_r * t[i] + (comb_est - t[i] * (mu_r + mu_f) - z0) * rho**2
            # remove from consideration any particles that already hit the bound
            tx[i] *= abs(x) < bound[i]

            # (6/23/2015) New method for computing p_know, p_remember, and confidences
            # simultaneously using cumulative bivariate normal integral.
            # The idea is to:
            #   1. model the bivariate distribution of (r,r+f) particle locations 'deltaT'
            #   seconds after the old/new decision
            #   2. use the multivariate normal integral function (stats.mvn.mvnun) to compute
            #   the probability that the particle falls into any of the relevant regions
            #   defined by the constant "r" and "conf" bounds

            ########################################################################
            # compute the parameters of the bivariate distribution of particle
            # locations deltaT seconds after old/new decision
            mu_r_delta = mu_r_cond + mu_r * deltaT
            mu_comb_delta = (mu_r + mu_f) * deltaT + comb_est
            s2_r_delta = s_r_cond**2 + 2 * d_r * deltaT
            s2_f_delta = s_f_cond**2 + 2 * d_f * deltaT
            s2_comb_delta = s2_r_delta + s2_f_delta
            # s2_comb_delta = 2*deltaT*(d_r+d_f);
            # s2_comb_delta = s_r_cond**2+s_f_cond**2+2*deltaT*(d_r+d_f);
            cov_delta = s2_r_delta
            mu_mvn = pl.array([mu_r_delta, mu_comb_delta])
            sigma_mvn = pl.array([[s2_r_delta, cov_delta], [cov_delta, s2_comb_delta]])
            mvn_dist = stats.multivariate_normal(mean=mu_mvn, cov=sigma_mvn, allow_singular=True)
            ########################################################################
            # Test Code:
            # if(t[i]>0.5):
            # slim = 3.0;
            # xaxis = linspace(-slim,slim,200);
            # xsup,ysup = meshgrid(xaxis,pl.flipud(xaxis));
            # supp = pl.vstack([xsup.flatten(),ysup.flatten()]).T;
            # z = stats.multivariate_normal.pdf(supp,mu_mvn,sigma_mvn);
            # figure(); imshow(reshape(z,shape(xsup)),cmap=cm.gray,extent=[-slim,slim,-slim,slim]);
            # vlines(r_bound,-slim,slim,colors='g');
            # hlines(c,-3,3,colors='r');
            # 1/0
            ########################################################################
            for j in range(1, len(clims)):
                # remember that clims contains the bin edges in descending order
                KLL = pl.array([-INF_PROXY, clims[j]])
                KUL = pl.array([r_bound, clims[j - 1]])
                RLL = pl.array([r_bound, clims[j]])
                RUL = pl.array([INF_PROXY, clims[j - 1]])
                p_know_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, KLL, KUL)
                p_rem_conf[j - 1, i] = p_old[i] * get_rect_prob(mvn_dist, RLL, RUL)

        # compute the marginal distributions for remember and know (i.e., across all confidence levels)
        p_remember = p_rem_conf.sum(0)
        p_know = p_know_conf.sum(0)
        return p_rem_conf, p_know_conf, p_new, t


params_est = DISPClassicDDM.Params(
    0.9984, 0.002, 0.3035, 0.0037, 0.3736, 0.0585, 0.001, -0.1306, -0.0142, -0.2438, 0.5859, 0.5126
)

param_bounds = DISPClassicDDM.Params(
    (0.0, 1.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (EPS, 1.0),
    (EPS, 1.0),
    (0.05, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (EPS, 2.0),
    (0, 0.5),
)
