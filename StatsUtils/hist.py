import typing
from typing import Callable, Tuple

import numpy as np
import scipy.integrate
import scipy.interpolate
from scipy.interpolate._bsplines import BSpline
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from .dists.empirical import getEmpiricalPDF
from .utils import differentiate, edges2Middles, empiricalPDFIntoEmpiricalCDF, renormalizeCDF, renormalizePDF, NumT


def getPPFBasedGridFromPPFPoints(bounds: Tuple[NumT, NumT], count: int, dBasedCDFY: np.ndarray, dBasedCDFX: np.ndarray, k: int = 3) -> np.ndarray:
	"""Gets a regular grid based on PPF"""
	interpolatedPPF = scipy.interpolate.make_interp_spline(dBasedCDFY, dBasedCDFX, k=k)
	binEdges = getPPFBasedGrid((np.min(dBasedCDFY), np.max(dBasedCDFY)), count + 1, interpolatedPPF)
	return binEdges


def getPPFBasedGrid(bounds: Tuple[NumT, NumT], count: int, interpolatedPPF: BSpline) -> np.ndarray:
	"""Gets a regular grid based on PPF"""
	yGrid = np.linspace(*bounds, count)
	xGrid = interpolatedPPF(yGrid)
	return xGrid


def getDerivativeBasedHistogramBins(dist: rv_continuous_frozen, countOfBins: int, pValue: float = 0.0001) -> np.ndarray:
	miab = dist.interval(1.0 - pValue)
	interpX = np.linspace(*miab, 100)
	binEdges = getPDFDerivativeBasedBinsFromEmpiricalHistBasedPDF(dist.pdf, interpX, countOfBins, miab)
	return binEdges


def genDerivativeBasedPDFFromEmpiricalHistBasedPDF(pdf: Callable, interpX: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Constructs PDF of a distribution generated from function derivative"""
	assert not np.any(np.isnan(interpX))
	absDeriv = np.abs(np.diff(pdf(interpX)))
	assert not np.any(np.isnan(absDeriv))
	xs = edges2Middles(interpX)
	res = absDeriv
	res = res / scipy.integrate.simpson(res, xs)
	return xs, res


def getDerivativeBasedGridFromEmpiricalHistBasedPDF(pdf: Callable, interpX: np.ndarray, binsCount: int) -> np.ndarray:
	dBasedPDFX, dBasedPDFY = genDerivativeBasedPDFFromEmpiricalHistBasedPDF(pdf, interpX)

	assert not np.any(np.isnan(dBasedPDFY))

	dBasedCDFX, dBasedCDFY = empiricalPDFIntoEmpiricalCDF(dBasedPDFX, dBasedPDFY)
	assert not np.any(np.isnan(dBasedCDFY))

	xGrid = getPPFBasedGridFromPPFPoints((np.min(dBasedCDFY), np.max(dBasedCDFY)), binsCount + 1, dBasedCDFY, dBasedCDFX, k=3)
	return xGrid


def getPDFDerivativeBasedBinsFromEmpiricalHistBasedPDF(pdf: Callable, interpX: np.ndarray, binsCount: int, miab: Tuple[NumT, NumT]) -> np.ndarray:
	xGrid = getDerivativeBasedGridFromEmpiricalHistBasedPDF(pdf, interpX, binsCount)

	xGrid[0] = min(miab[0], xGrid[0])
	xGrid[-1] = max(miab[-1], xGrid[-1])

	return xGrid


def getPDFDerivativeBasedBinsFromPoints(points, binsCount):
	xs, ys = getEmpiricalPDF(points)
	dBasedPDFX, dBasedPDFY = differentiate(xs, ys)
	dBasedPDFY = np.abs(dBasedPDFY)

	assert not np.any(np.isnan(dBasedPDFY))

	dBasedPDFY = renormalizePDF(dBasedPDFX, dBasedPDFY)
	dBasedCDFX, dBasedCDFY = empiricalPDFIntoEmpiricalCDF(dBasedPDFX, dBasedPDFY)
	assert not np.any(np.isnan(dBasedCDFY))

	binEdges = getPPFBasedGridFromPPFPoints((np.min(dBasedCDFY), np.max(dBasedCDFY)), binsCount + 1, dBasedCDFY, dBasedCDFX, k=3)
	return binEdges
