import typing
from collections import defaultdict
from functools import partial

import numpy as np
import scipy.signal
import scipy.stats
from numpy import float32, float64, int32, ndarray
from scipy.integrate import simpson
from scipy.optimize import curve_fit, minimize
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multivariate_normal_frozen

from . import core
from .utils import BoundsT, MultivariateBoundsT, NumT, computeFunctionOnGrid


def getDistFitterFunc(distCtor):
	def fitNomalFromPoints(points):
		params = distCtor.fit(points)
		return distCtor(*params)

	return fitNomalFromPoints
