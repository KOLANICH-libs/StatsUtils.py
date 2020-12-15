import typing

import numpy as np
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_frozen

from .utils import NumT, getCountOfBinsForPoints


def getHistogramBins(self: rv_continuous_frozen, bins: int, pValue: NumT = 0.0001) -> np.ndarray:
	from .hist import getDerivativeBasedHistogramBins  # pylint:disable=import-outside-toplevel

	return getDerivativeBasedHistogramBins(self, bins, pValue=pValue)


rv_continuous.getHistogramBins = rv_frozen.getHistogramBins = getHistogramBins


def adaptiveHistogram(self: rv_continuous_frozen, points: np.ndarray, pValue: NumT = 0.0001, bins: int = None, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:
	"""Computes a histogram with bins situated so that each bin got kinda equal probability integral increment"""

	if bins is None:
		bins = getCountOfBinsForPoints(points)

	binEdges = self.getHistogramBins(bins, pValue=pValue)

	freqs, binEdges = np.histogram(points, bins=binEdges, **kwargs)
	return binEdges, freqs


rv_continuous.adaptiveHistogram = rv_frozen.adaptiveHistogram = adaptiveHistogram


@classmethod
def fitPDFMLE(cls, ys: np.ndarray, sy: np.ndarray, method=None):
	raise NotImplementedError


rv_continuous.fitPDFMLE = rv_frozen.fitPDFMLE = fitPDFMLE


@classmethod
def fitPointsMLE(cls, points: np.ndarray, method=None) -> (rv_frozen, typing.Any):
	raise NotImplementedError


rv_continuous.fitPointsMLE = fitPointsMLE


def KLDiv(self: rv_continuous_frozen, xs: np.ndarray, sx: np.ndarray) -> np.ndarray:
	logpdf = self.logpdf(xs)
	nans = np.isnan(logpdf)
	assert not np.any(nans), np.argwhere(nans).T
	del nans
	return -np.sum(logpdf * sx)


rv_continuous.KLDiv = rv_frozen.KLDiv = KLDiv


def nnlf(self: rv_continuous_frozen, points: np.ndarray) -> float:
	return -np.sum(self.logpdf(points))


rv_continuous.nnlf = rv_frozen.nnlf = nnlf
