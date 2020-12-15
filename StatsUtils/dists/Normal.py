from functools import partial

import numpy as np
import scipy.optimize
from scipy.stats._distn_infrastructure import rv_frozen

from .. import core
from ..utils import calcRealMean, calcVarianceGivenMeanUnbiased, renormalizePDF

__all__ = ()


def fitPDFMLE(ys: np.ndarray, sy: np.ndarray, method=None) -> (rv_frozen, None):
	assert len(ys) == len(sy)
	sy = renormalizePDF(ys, sy)
	m = calcRealMean(ys, sy)
	d = calcVarianceGivenMeanUnbiased(m, ys, sy)
	return scipy.stats.norm(loc=m, scale=np.sqrt(d)), None


scipy.stats.norm.fitPDFMLE = fitPDFMLE


def fitPointsMLE(cls, points: np.ndarray, method=None) -> (rv_frozen, None):
	loc, scale = cls.fit(points)
	return cls(loc=loc, scale=scale), None


scipy.stats.norm.fitPointsMLE = partial(fitPointsMLE, scipy.stats.norm)


def var_var(self, count: int) -> float:
	return self.kurtosis() / count


scipy.stats.norm.var_var = var_var


def mode(self) -> float:
	return self.mean()


scipy.stats.norm.mode = mode
