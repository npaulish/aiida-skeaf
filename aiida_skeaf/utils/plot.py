#!/usr/bin/env python
"""Functions to plot output results."""
import matplotlib.pyplot as plt
import numpy as np

from aiida import orm
from aiida.common import AttributeDict

from aiida_skeaf.calculations import SkeafCalculation
from aiida_skeaf.workflows import SkeafWorkChain


def complement_color(r, g, b):
    """Get complement color.

    https://stackoverflow.com/questions/40233986/python-is-there-a-function-or-formula-to-find-the-complementary-colour-of-a-rgb
    :param r: _description_
    :type r: _type_
    :param g: _description_
    :type g: _type_
    :param b: _description_
    :type b: _type_
    """

    def hilo(a, b, c):
        # Sum of the min & max of (a, b, c)
        if c < b:
            b, c = c, b
        if b < a:
            a, b = b, a
        if c < b:
            b, c = c, b
        return a + c

    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))


def plot_xy(  # pylint: disable=too-many-arguments
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str = None,
    xlabel: str = None,
    ylabel: str = None,
    title: str = "Frequency vs angle",
    ax: plt.Axes = None,
    show_plot: bool = False,
    show_legend: bool = True,
    x_offset: float = 0.0,
    scale_x: float = 1.0,
    invert_x: bool = False,
    **kwargs,
) -> None:
    """Plot raw data.

    :param x: _description_
    :type x: np.ndarray
    :param y: _description_
    :type y: np.ndarray
    :param label: _description_, defaults to None
    :type label: str, optional
    :param ax: If provided reuse this matplotlib axes, defaults to None
    :type ax: matplotlib.pyplot.Axes, optional
    :param show_plot: _description_, defaults to False
    :type show_plot: bool, optional
    :param show_legend: _description_, defaults to True
    :type show_legend: bool, optional
    :param x_offset: _description_, defaults to 0.0
    :type x_offset: float, optional
    :param scale_x: _description_, defaults to 1.0
    :type scale_x: float, optional
    """

    if ax is None:
        _, ax = plt.subplots(1, 1)
        show_plot = True

    if invert_x:
        ax.xaxis.set_inverted(True)

    ax.scatter(
        scale_x * x + x_offset,
        y,
        label=label,
        **kwargs,
    )

    xlabel = kwargs.get("xlabel", xlabel)
    ylabel = kwargs.get("ylabel", ylabel)
    title = kwargs.get("title", title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend:
        ax.legend()

    if show_plot:
        plt.show()


def plot_frequency(
    frequency: orm.ArrayData,
    x: str = "phi",
    y: str = "freq",
    *,
    multiply_cosine: bool = False,
    **kwargs,
) -> None:
    """Plot ``SkeafCalculation.outputs.frequency`` array.

    :param frequency: ``SkeafCalculation.outputs.frequency``
    :type frequency: orm.ArrayData
    :param x: the array to be used as x data, defaults to "phi"
    :type x: str, optional
    :param y: the array to be used as y data, defaults to "freq"
    :type y: str, optional
    :param multiply_cosine: multiply y array with cos(theta), theta from z axis
    should be a horizontal line for cylindrical Fermi surface.
    :type multiply_cosine: bool, optional
    """
    if any(_ not in frequency.get_arraynames() for _ in [x, y]):
        raise ValueError(f"valid x and y are {frequency.get_arraynames()}")

    x_array = frequency.get_array(x)
    y_array = frequency.get_array(y)

    if multiply_cosine:
        y_array *= np.cos(x_array / 180 * np.pi)

    header = frequency.base.attributes.all["header"].strip().split(",")

    xlabel = [_ for _ in header if x in _.lower()][0]
    ylabel = [_ for _ in header if y in _.lower()][0]

    plot_xy(x_array, y_array, xlabel=xlabel, ylabel=ylabel, **kwargs)


def plot_frequency_calc(
    calc: SkeafCalculation,
    **kwargs,
) -> None:
    """Plot SKEAF output frequency vs angle.

    :param calc: A finished ``SkeafCalculation``
    :type calc: SkeafCalculation
    """
    frequency: orm.ArrayData = calc.outputs.frequency

    plot_frequency(frequency=frequency, **kwargs)


def plot_frequency_workchain(
    wkchain: SkeafWorkChain,
    **kwargs,
) -> None:
    """Plot SKEAF output frequency vs angle.

    :param wkchain: A finished ``SkeafWorkChain``
    :type wkchain: SkeafWorkChain
    """
    show_plot = False
    if "ax" not in kwargs:
        _, ax = plt.subplots()
        show_plot = True
    else:
        ax = kwargs.pop("ax")

    skeaf_outputs: AttributeDict = wkchain.outputs.skeaf
    # Sort by band index
    skeaf_outputs = dict(sorted(skeaf_outputs.items(), key=lambda _: _[0]))

    for band_idx, skeaf_out in skeaf_outputs.items():
        frequency: orm.ArrayData = skeaf_out.frequency

        plot_frequency(frequency=frequency, label=band_idx, ax=ax, **kwargs)

    ax.set_title(f"{wkchain.process_label}<{wkchain.pk}>")

    if show_plot:
        plt.show()
