import typing
from itertools import product

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.stats._multivariate import multivariate_normal_frozen

from .. import core
from ..utils import MultivariateBoundsT, NumT, getRotMatrix, rotMat2x2ToAngle


class MultivariateGaussian:
	"""A class representing a statistical distribution, but which functionality is not a part of `scipy.stats` currently. It's a shame!"""

	DIST = scipy.stats.multivariate_normal

	__slots__ = ("mode", "dist", "variances", "angles", "covMat", "rotMat", "stds")

	def __init__(self, *, mode, variances=None, stds=None, covMat=None, rotMat=None, angles=None) -> None:
		if covMat is not None:
			if rotMat is None and variances is None:
				variances, rotMat = decompose(covMat)

		if rotMat is not None:
			if angles is None:
				angles = rotMat2x2ToAngle(rotMat)

		if stds is None and variances is not None:
			stds = np.sqrt(variances)

		if covMat is None:
			if stds is not None:
				if rotMat is not None:
					covMat = self.__class__.covMatFromRotMatrixAndStds(stds, rotMat)
				elif angles is not None:
					rotMat = getRotMatrix(angles)
					covMat = self.__class__.covMatFromAnglesAndStds(stds, angles)
				else:
					raise NotImplementedError("You must also specify either `angles` or `rotMat`")
			else:
				raise NotImplementedError("You must specify either `variances` or `stds` or `covMat`")

		self.variances = variances
		self.stds = stds
		self.angles = angles
		self.rotMat = rotMat
		self.covMat = covMat
		self.mode = mode
		self.dist = self.__class__.DIST(mean=mode, cov=covMat)  # https://github.com/scipy/scipy/issues/15675

	def fitPDFMLE(self, ys: np.ndarray, sy: np.ndarray):
		raise NotImplementedError

	@classmethod
	def fitPointsMLE(cls, points: np.ndarray) -> None:
		"""https://github.com/scipy/scipy/issues/15676"""
		m = np.mean(points, axis=0)
		ms = points - m
		c = ms.T @ ms / (len(points) - 1)
		return cls(mode=m, covMat=c)

	@classmethod
	def unrotatedCovMatrix(cls, sigmas):
		return np.diag(sigmas ** 2)

	@classmethod
	def covMatFromAnglesAndStds(cls, sigmas, angles):
		return cls.covMatFromRotMatrixAndStds(sigmas, getRotMatrix(angles))

	@classmethod
	def covMatFromRotMatrixAndStds(cls, sigmas, r):
		return r @ cls.unrotatedCovMatrix(sigmas) @ r.T

	def computeBounds(self, pValue: NumT = 0.001, ns: NumT = None) -> MultivariateBoundsT:
		"""https://github.com/scipy/scipy/issues/15677"""

		if ns is None:
			nst = []
			for s in self.stds:
				d = scipy.stats.norm(loc=0, scale=s)
				ns = d.interval(1.0 - pValue)
				nst.append(ns)
		else:
			nst = self.stds
			for n, s in zip(ns, self.stds):
				ns = n * s
				nst.append([-ns, +ns])
		nts = np.array(nst)

		bools = (0, 1)
		bounds = []
		for el in product(*([bools] * len(self.stds))):
			bounds.append([nts[0][el[0]], nts[1][el[1]]])
		boundsTransformed = (np.array(bounds) @ self.rotMat).T
		bounds = []

		for i in range(boundsTransformed.shape[0]):
			bb = boundsTransformed[i] + self.mode[i]
			bounds.append((np.min(bb), np.max(bb)))
		bounds = np.array(bounds)
		return bounds


def decompose(c: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
	S, e = scipy.linalg.eig(c)
	return np.real(S), np.real(e)


def fitSingleJointDistribution(points: np.ndarray) -> typing.Tuple[typing.Mapping, multivariate_normal_frozen, MultivariateBoundsT]:
	d = MultivariateGaussian.fitPointsMLE(points)
	bounds = d.computeBounds(ns=10)

	return {"m": d.mode.tolist(), "v": d.variances.tolist(), "r": d.angles}, bounds
