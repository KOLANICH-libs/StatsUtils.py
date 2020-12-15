import typing
from functools import partial

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.stats._continuous_distns import truncnorm_gen
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from .. import core
from ..utils import BoundsT, NumT, calcVarianceGivenMeanMaybeBiased
from ..utils.optimize import optimize

__all__ = ()


def constructTruncNorm(cls: truncnorm_gen, bounds: BoundsT, mode: ndarray, std: NumT = None, variance: NumT = None) -> rv_continuous_frozen:
	assert not np.any(np.isnan(mode))
	if std is None:
		std = np.sqrt(variance)
	else:
		if variance is None:
			variance = std * std

	assert not np.isnan(variance)
	assert not np.isnan(std)

	res = cls(a=(bounds[0] - mode) / std, b=(bounds[1] - mode) / std, loc=mode, scale=std)
	res.bounds = bounds
	res._mode = mode
	# res.std = std
	res.variance = variance

	return res


scipy.stats.truncnorm.altCtor = partial(constructTruncNorm, scipy.stats.truncnorm)


def _truncNormKL(cls: truncnorm_gen, bounds: BoundsT, x: ndarray, y: ndarray, m: NumT, sigma: NumT) -> float:
	assert len(x) == len(y)
	d = cls.altCtor(mode=m, std=sigma, bounds=bounds)
	return d.KLDiv(x, y)


scipy.stats.truncnorm._truncNormKL = partial(_truncNormKL, scipy.stats.truncnorm)


def _truncNormPdf(cls: truncnorm_gen, bounds: BoundsT, x: ndarray, m: NumT, sigma: NumT) -> ndarray:
	d = cls.altCtor(mode=m, std=sigma, bounds=bounds)
	return d.pdf(x)


scipy.stats.truncnorm._truncNormPdf = partial(_truncNormPdf, scipy.stats.truncnorm)


def _optimizeTruncNormParams(cls: truncnorm_gen, method: typing.Type[Yabox], objective: partial, m: NumT, sigma: NumT, restParams: typing.Iterable[typing.Any], bounds: BoundsT) -> typing.Tuple[typing.Tuple[NumT, NumT], None]:
	return optimize(
		method, objective, (bounds, *restParams), [m, sigma], [bounds, (sigma / 2, sigma * 2)],
		{
			"m": cls.altCtor(mode=m, std=m, bounds=bounds),
			"sigma": cls.altCtor(mode=sigma, std=sigma / 2, bounds=[sigma / 2, sigma * 2])
		}
	)


scipy.stats.truncnorm._optimizeTruncNormParams = partial(_optimizeTruncNormParams, scipy.stats.truncnorm)


def fitPDFMLE(cls: truncnorm_gen, x: ndarray, y: ndarray, bounds: BoundsT = (0, 1), method=differential_evolution) -> typing.Tuple[rv_continuous_frozen, NumT]:
	assert len(x) == len(y)
	m = x[np.argmax(y)]
	d = calcVarianceGivenMeanMaybeBiased(m, x, y)
	sigma = np.sqrt(d)
	errors = None
	# preliminary curve fitting
	(m, sigma), oCov = curve_fit(partial(cls._truncNormPdf, bounds), x, y, p0=[m, sigma])
	# MLE estimation by KL minimization
	#res = minimize(partial(cls._truncNormKL, bounds, x, y), x0=[m, sigma], method="L-BFGS-B", bounds=(bounds, (sigma / 3, sigma * 3)))
	(m, sigma), error = cls._optimizeTruncNormParams(method, cls._truncNormKL, m, sigma, (x, y), bounds)

	errors = np.sum(np.abs(cls._truncNormPdf(bounds, x, m, sigma) - y))
	d = sigma ** 2
	dist = cls.altCtor(mode=m, std=sigma, bounds=bounds)
	return dist, errors


scipy.stats.truncnorm.fitPDFMLE = partial(fitPDFMLE, scipy.stats.truncnorm)


def _truncNormNNLL(cls: truncnorm_gen, bounds: BoundsT, points: ndarray, m: NumT, sigma: NumT) -> NumT:
	d = scipy.stats.truncnorm.altCtor(mode=m, std=sigma, bounds=bounds)
	return d.nnlf(points)


scipy.stats.truncnorm._truncNormNNLL = partial(_truncNormNNLL, scipy.stats.truncnorm)


def fitPointsMLE(cls: truncnorm_gen, points: ndarray, bounds: BoundsT = (0, 1), method=differential_evolution) -> None:
	error = None
	m, sigma = scipy.stats.norm.fit(points)  # initial approximation
	(m, sigma), error = cls._optimizeTruncNormParams(method, cls._truncNormNNLL, m, sigma, (points,), bounds)

	return cls.altCtor(mode=m, std=sigma, bounds=bounds), error


scipy.stats.truncnorm.fitPointsMLE = partial(fitPointsMLE, scipy.stats.truncnorm)
