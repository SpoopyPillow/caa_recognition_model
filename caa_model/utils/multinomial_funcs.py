import pylab as pl
import scipy as sp
from scipy.special import gamma, gammaln

EPS = 1e-10  # a very small value (used for numerical stability)


def multinom_loglike(x, n, p):
    """
    where x is a k-length vector representing the number of observed events in
    each category, n is a scalar representing the number of trials, and p is a
    k-length vector representing the proportion parameter for each category

    returns the log likelihood of obtaining x events in each category
    """
    # Clip the input values to lie within a valid range
    x = pl.clip(x, EPS, None)
    p = pl.clip(p, EPS, None)

    return gammaln(n + 1) - pl.sum(gammaln(x + 1)) + pl.sum(x * pl.log(p))


def chi_square_gof(x, n, p):
    """
    where x is a k-length vector representing the number of observed events in
    each category, n is a scalar representing the number of trials, and p is a
    k-length vector representing the proportion parameter for each category

    returns the associated chi-squared statistic
    """
    p = pl.clip(p, EPS, None)
    return sum((x - n * p) ** 2 / (n * p))


def get_rect_prob(rv, low, high):
    """
    Get probability of a random variable in a rectangle.
    """
    pts = pl.array([[high[0], high[1]], [low[0], high[1]], [high[0], low[1]], [low[0], low[1]]])
    p = rv.cdf(pts)
    return p[0] - p[1] - p[2] + p[3]
