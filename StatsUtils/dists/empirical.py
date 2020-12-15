import typing

import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.stats import rv_continuous

from ..core import *
from ..utils import edges2Middles, renormalizeCDF, renormalizePDF

def sortWithOptionalDeduplicationAndWeightsComputation(points: np.ndarray, dedup: bool) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
	if dedup:
		return np.unique(points, return_counts=True)
	else:
		return np.sort(points), None

def getEmpiricalCDF(points: np.ndarray, k=3):
	return _getEmpiricalCDFFromSortedPoints(np.sort(points), k=k)


def _getEmpiricalCDFFromSortedPoints(sortedPoints: np.ndarray, k=3):
	return sortedPoints, np.linspace(0, 1, len(sortedPoints))


def getEmpiricalPPF(points: np.ndarray, k=3):
	return _getEmpiricalPPFFromSortedPoints(np.sort(points), k=k)


def _getEmpiricalPPFFromSortedPoints(sortedPoints: np.ndarray, k=3):
	return np.linspace(0, 1, len(sortedPoints)), sortedPoints


def getEmpiricalPDF(points: np.ndarray, k=3, dedup: bool=False):
	return _getEmpiricalPDFFromSortedPoints(*sortWithOptionalDeduplicationAndWeightsComputation(points, dedup=dedup), k=k)

def _getEmpiricalPDFFromSortedPoints(sortedPoints: np.ndarray, weights: typing.Optional[np.ndarray], k=3):
	cdfX = sortedPoints
	pdfX = edges2Middles(cdfX)
	cdfXDeltas = np.diff(cdfX)
	if weights is not None:
		weights = edges2Middles(weights)
		partFun = np.sum(weights)
	else:
		partFun = len(pdfX)

	pdfY = (1 / partFun) / cdfXDeltas
	if weights is not None:
		pdfY *= weights

	pdfY = preprocessEmpiricalPDF(pdfY)
	pdfY = renormalizePDF(pdfX, pdfY)
	return pdfX, pdfY


def getEmpiricalCDFPDF(points: np.ndarray, k=3, dedup: bool=False):
	cdfX, weights = sortWithOptionalDeduplicationAndWeightsComputation(points, dedup=dedup)

	cdf = _getEmpiricalCDFFromSortedPoints(cdfX, k=3)
	pdf = _getEmpiricalPDFFromSortedPoints(cdfX, weights, k=3)
	return cdf, pdf


def preprocessEmpiricalPDF(pdfY: np.ndarray, gaussianWidth: float = 0.025):
	gWSize = int(round(len(pdfY) * gaussianWidth))
	w = scipy.signal.windows.gaussian(gWSize * 3, gWSize)

	mfWSize = int(round(np.sqrt(len(pdfY))))
	mfWSize -= (mfWSize + 1) % 2

	for _ in range(2):
		pdfY = scipy.signal.medfilt(pdfY, mfWSize)
		pdfY = scipy.signal.convolve(pdfY, w, mode="same")
	return pdfY


class rv_empirical(rv_continuous):
	__slots__ = ("cdfX", "pdfX", "weights", "pdfY", "pdfInterp", "cdfInterp", "ppfInterp", "k")

	_support_mask = rv_continuous._support_mask

	def __init__(self, *, points: np.ndarray = None, cdfX: np.ndarray = None, weights: typing.Optional[np.ndarray] = None, cdfY: np.ndarray = None, pdfX: np.ndarray = None, pdfY: np.ndarray = None, pdfInterp: typing.Callable = None, cdfInterp: typing.Callable = None, k=3, dedup: bool=False, **kwargs):
		self.k = k
		self.cdfX = cdfX
		self.weights = weights
		self.pdfX = pdfX
		self.pdfY = pdfY

		ppfInterp = None

		if points is not None:
			if self.pdfX is not None or self.pdfY is not None or self.cdfX is not None or cdfY is not None:
				raise ValueError("You have provided points, PDF and CDF will be replaced by the one computed from points")

			self.cdfX, self.weights = sortWithOptionalDeduplicationAndWeightsComputation(points, dedup)
			self._computePDF()

		if pdfInterp is None:
			if self.pdfX is not None and self.pdfY is not None:
				pdfInterp = scipy.interpolate.make_interp_spline(self.pdfX, self.pdfY, k=self.k)

		if self.cdfX is not None and cdfY is not None:
			if cdfInterp is None:
				cdfInterp = scipy.interpolate.make_interp_spline(self.cdfX, cdfY, k=self.k)

			if ppfInterp is None:
				ppfInterp = scipy.interpolate.make_interp_spline(cdfY, self.cdfX, k=self.k)

		if cdfInterp is None:
			cdfInterp = pdfInterp.antiderivative()

		if pdfInterp is None:
			pdfInterp = cdfInterp.derivative()

		if ppfInterp is None:
			if self.cdfX is not None and cdfInterp is not None:
				ppfInterp = scipy.interpolate.make_interp_spline(cdfInterp(self.cdfX), self.cdfX, k=self.k)

		if cdfInterp is None:
			raise ValueError("Not enough data to compute cdfInterp")

		if pdfInterp is None:
			raise ValueError("Not enough data to compute pdfInterp")

		if ppfInterp is None:
			raise ValueError("Not enough data to compute ppfInterp")

		self.pdfInterp = pdfInterp
		self.cdfInterp = cdfInterp
		self.ppfInterp = ppfInterp

		# Set support
		kwargs["a"] = self.a = self.ppfInterp(0)
		kwargs["b"] = self.b = self.ppfInterp(1)
		super().__init__(**kwargs)

	def _computePDF(self):
		# (cdfX, cdfY), (pdfX, self.pdfY) = getEmpiricalCDFPDF(points)
		(self.pdfX, self.pdfY) = _getEmpiricalPDFFromSortedPoints(self.cdfX, self.weights, k=self.k)

	def _pdf(self, x):
		# return np.abs(self.pdfInterp(x))
		return self.pdfInterp(x)

	def _cdf(self, x):
		return self.cdfInterp(x)

	def _ppf(self, x):
		return self.ppfInterp(x)

	def _updated_ctor_param(self):
		dct = super(rv_histogram, self)._updated_ctor_param()
		dct["pdfInterp"] = self.pdfInterp
		dct["cdfInterp"] = self.cdfInterp
		dct["ppfInterp"] = self.ppfInterp
		return dct
