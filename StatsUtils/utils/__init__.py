import typing

import numpy as np
import scipy.linalg
import scipy.signal
import scipy.stats
from numpy import float32, float64, int32, ndarray
from scipy.integrate import simpson
from scipy.spatial.transform import Rotation as R
from scipy.stats._distn_infrastructure import rv_frozen

NumT = typing.Union[int, int32, float, float32, float64]
RangeSpecT = typing.Tuple[NumT, NumT, int]
BoundsT = typing.Tuple[NumT, NumT]
MultivariateBoundsT = typing.Tuple[BoundsT, ...]


def rotMat2x2ToAngle(rotMat: ndarray) -> float:
	angl = np.arctan2(rotMat[1, 0], rotMat[0, 0])
	if angl > np.pi / 2:
		angl = np.pi - angl

	if angl < -np.pi / 2:
		angl = np.pi + angl

	return angl


def edges2Middles(x: np.ndarray) -> np.ndarray:
	return (x[1:] + x[:-1]) / 2


def edges2Widths(x: np.ndarray) -> np.ndarray:
	return x[1:] - x[:-1]


def renormalizePDF(x, pdf):
	return pdf / simpson(pdf, x)


def renormalizeCDF(cdf):
	return cdf / cdf[-1]


def calcError(x: ndarray, s: ndarray, dist: rv_frozen) -> NumT:
	return simpson(np.abs(s - dist.pdf(x)) * x, x)


def calcRealMean(x: ndarray, s: ndarray) -> NumT:
	return simpson(x * s, x)


def calcVarianceGivenMeanUnbiased(m: NumT, x: ndarray, s: ndarray) -> NumT:
	msq = m * m
	sqm = simpson(x * x * s, x)
	assert sqm > msq, (sqm, msq)
	return sqm - msq


def calcVarianceGivenMeanMaybeBiased(m: NumT, x: ndarray, s: ndarray) -> NumT:
	xc = x - m
	return simpson(xc * xc * s, x)


def integrateParametric(x: ndarray, y: ndarray) -> (ndarray, ndarray):
	"""Integrates PDF in a form of 2 arrays to get CDF"""
	return edges2Middles(x), scipy.integrate.cumulative_trapezoid(y, x)


empiricalPDFIntoEmpiricalCDF = integrateParametric


def differentiate(x: ndarray, y: ndarray) -> (ndarray, ndarray):
	return edges2Middles(x), np.diff(y) / np.diff(x)


def getCountOfBinsForPoints(points):
	binEdges = np.histogram_bin_edges(points, bins="auto")
	countOfBins = len(binEdges) - 1
	return countOfBins


def analyticalPDFFromMultipleDists(dists, x):
	res = np.zeros(x.shape)
	for d in dists:
		res += d.pdf(x)
	res /= len(dists)
	return res


def computeFunctionOnGrid(f: typing.Callable, xR: RangeSpecT, yR: RangeSpecT) -> typing.Tuple[ndarray, ndarray, ndarray]:
	xs = np.linspace(*xR, dtype=np.float64)
	ys = np.linspace(*yR, dtype=np.float64)
	dd = f(np.dstack(np.meshgrid(xs, ys)))
	return xs, ys, dd


def getRotMatrix(a):
	return R.from_euler("Z", np.array([a])).as_matrix()[0, :-1, :-1]
