import typing

import numpy as np

from .dists.MultivariateGaussian import MultivariateGaussian
from .utils import NumT, computeFunctionOnGrid, edges2Middles, edges2Widths


def plotPrecomputedHistogram(*, edges=None, freqs=None, middles=None, widths=None, **kwargs) -> None:
	from matplotlib import pyplot as plt

	if middles is None:
		middles = edges2Middles(edges)

	if widths is None:
		widths = edges2Widths(edges)

	return plt.bar(middles, freqs, width=widths, **kwargs)


def plotFunc(f, xR, yR, ax, **kwargs):
	"""Utility function to plot 2D functions"""
	xs, ys, dd = computeFunctionOnGrid(f, xR, yR)
	return ax.pcolormesh(xs, ys, dd, shading="auto", **kwargs)


def seabornJointPlotWithGaussian(points: ndarray, alpha: NumT = 0.2, resolution: typing.Tuple[int, int] = (150, 150)) -> typing.Tuple["seaborn.axisgrid.JointGrid", MultivariateGaussian, np.ndarray]:
	import seaborn  # pylint:disable=import-outside-toplevel

	d = MultivariateGaussian.fitPointsMLE(points)
	bounds = d.computeBounds(pValue=0.01)
	jp = seaborn.jointplot(x=points[:, 0], y=points[:, 1], kind="kde", joint_kws={"alpha": alpha}, fill=True)
	xr, yr = bounds
	plotFunc(d.dist.pdf, (*xr, resolution[0]), (*yr, resolution[1]), jp.ax_joint, zorder=-1)
	return jp, d, bounds
