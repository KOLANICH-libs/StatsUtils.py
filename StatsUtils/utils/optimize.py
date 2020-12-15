"""A small abstraction layer around minimizers.

Suppports both some built-into scipy and UniOpt ones.
"""

import sys
import typing
from functools import partial

from numpy import ndarray
from scipy.optimize import curve_fit, differential_evolution, minimize

from . import NumT


def _funcDictParams(f, fixedParams, params: typing.Mapping[str, float]):
	return f(*fixedParams, **params)


def _funcForUniOpt(f, fixedParams, params: typing.Mapping[str, float]):
	return (_funcDictParams(f, fixedParams, params), 0)


def _funcForMinimize(f, fixedParams, params: (NumT, NumT)) -> ndarray:
	return f(*fixedParams, *params)


def optimize(method, func, fixedParams, initialGuess, diffEvoBounds, uniOptParamsDists):
	if method is minimize:
		res = minimize(partial(_funcForMinimize, func, fixedParams), x0=initialGuess, method="BFGS")
		res, error = res["x"], res["fun"]
	elif method is differential_evolution:
		res = differential_evolution(
			partial(_funcForMinimize, func, fixedParams),
			x0=initialGuess,
			bounds=diffEvoBounds,
			args=(),
			tol=0.001,
			callback=None,
			disp=False,
			polish=True,
			vectorized=False
		)
		res, error = res["x"], res["fun"]
	else:

		UniOptCo = sys.modules.get("UniOpt.core", None)
		if UniOptCo is not None:
			if issubclass(method, UniOptCo.Optimizer.Optimizer):
				import UniOpt
				from UniOpt.core.Spec import HyperparamDefinition

				opt = method(
					partial(_funcForUniOpt, func, fixedParams),
					spaceSpec={
						k: HyperparamDefinition(float, v) for k, v in uniOptParamsDists.items()
					},
					iters=100,
					jobs=3,
					#pointsStorage: UniOpt.core.PointsStorage.PointsStorage = None
				)
				optRes = opt()
				res = tuple(optRes[k] for k in uniOptParamsDists)
				error = None
		else:
			raise ValueError("Unsupported method", method)

	return res, error
