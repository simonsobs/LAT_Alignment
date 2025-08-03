"""
Helper module with plotting functions for the trajectory module.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_by_ax_point(names: list[str] | NDArray[np.str_], x: NDArray[np.float64], dat: NDArray[np.float64], direction: NDArray[np.float64], missing: list[int], xax: str, xlab: str, title: str, plt_root: str):
    """
    Plot data with subplots organized such that each row is a different dimension
    and each collumn is a different physical location that was measured.
    The plot will be saved to `plt_root/{title.lower().replace(' ' , '_')}_{xax}.png`.

    Parameters
    ----------
    names : list[str] | NDArray[np.str_]
        The names of the physically measured locations.
    x : NDArray[np.float64]
        The x data to plot against.
        Should have shape `(npoint,)`.
    dat : NDArray[np.float64]
        The data to plot.
        Should have shape `(npoint, nloc, ndim)`.
    direction : NDArray[np.float64]
        The direction of motion at each data point.
        Should have shape `(npoint,)`.
    missing : list[int]
        A list of indices that will be flagged in the plot
        as only having partial data.
    xax : str
        A description of the x axis for the file name.
    xlab : str
        The x label to use in the plot.
    title : str
        The title of the plot.
    plt_root : str
        The directory to save plots to.
    """
    _, axs = plt.subplots(
        3,
        len(names),
        sharex=True,
        sharey=False,
        figsize=(24, 20),
        layout="constrained",
    )
    axs = np.reshape(np.array(axs), (int(3), len(names)))
    for i, point in enumerate(names):
        for j, dim in enumerate(["x", "y", "z"]):
            axs[j, i].scatter(
                x[direction == 0],
                dat[direction == 0, i, j],
                color="black",
                marker="o",
                alpha=0.5,
                label="Stationary",
            )
            axs[j, i].scatter(
                x[direction < 0],
                dat[direction < 0, i, j],
                color="blue",
                marker="x",
                alpha=0.5,
                label="Decreasing",
            )
            axs[j, i].scatter(
                x[direction > 0],
                dat[direction > 0, i, j],
                color="red",
                marker="+",
                alpha=0.5,
                label="Increasing",
            )
            axs[j, i].scatter(x[missing], dat[missing, i, j], color="gray", marker="1")
            axs[0, i].set_title(point)
            axs[-1, i].set_xlabel(xlab)
            axs[j, 0].set_ylabel(f"{dim} (mm)")
    axs[-1, 0].legend()
    plt.suptitle(title)
    plt.savefig(
        os.path.join(plt_root, f"{title.lower().replace(' ' , '_')}_{xax}.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_by_ax(x: NDArray[np.float64], dat: NDArray[np.float64], direction: NDArray[np.float64], missing: list[int], xax: str, xlab: str, ylab: str, title: str, plt_root: str):
    """
    Plot data with seperate subplots for each dimension. 
    The plot will be saved to `plt_root/{title.lower().replace(' ' , '_')}_{xax}.png`.

    Parameters
    ----------
    x : NDArray[np.float64]
        The x data to plot against.
        Should have shape `(npoint,)`.
    dat : NDArray[np.float64]
        The data to plot.
        Should have shape `(npoint, ndim)`.
    direction : NDArray[np.float64]
        The direction of motion at each data point.
        Should have shape `(npoint,)`.
    missing : list[int]
        A list of indices that will be flagged in the plot
        as only having partial data.
    xax : str
        A description of the x axis for the file name.
    xlab : str
        The x label to use in the plot.
    ylab : str
        The y label to use in the plot.
    title : str
        The title of the plot.
    plt_root : str
        The directory to save plots to.
    """
    _, axs = plt.subplots(3, 1, sharex=True)
    for i, dim in enumerate(["x", "y", "z"]):
        axs[i].scatter(
            x[direction == 0],
            dat[direction == 0, i],
            color="black",
            marker="o",
            alpha=0.25,
            label="Stationary",
        )
        axs[i].scatter(
            x[direction < 0],
            dat[direction < 0, i],
            color="blue",
            marker="x",
            alpha=0.25,
            label="Decreasing",
        )
        axs[i].scatter(
            x[direction > 0],
            dat[direction > 0, i],
            color="red",
            marker="+",
            alpha=0.25,
            label="Increasing",
        )
        axs[i].scatter(x[missing], dat[missing, i], color="gray", marker="1")
        axs[i].set_ylabel(f"{dim} {ylab}")
    axs[0].legend()
    axs[-1].set_xlabel(xlab)
    plt.suptitle(title)
    plt.savefig(
        os.path.join(plt_root, f"{title.lower().replace(' ' , '_')}_{xax}.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_all_ax(x: NDArray[np.float64], dat: NDArray[np.float64],  missing: list[int], xlab: str, ylab: str, title: str, plt_root: str):
    """
    Plot data with dimensions as seperate lines on the same plot. 
    This doesn't show the direction of motions on the plot.
    The plot will be saved to `plt_root/{title.lower().replace(' ' , '_')}.png`.

    Parameters
    ----------
    x : NDArray[np.float64]
        The x data to plot against.
        Should have shape `(npoint,)`.
    dat : NDArray[np.float64]
        The data to plot.
        Should have shape `(npoint, ndim)`.
    missing : list[int]
        A list of indices that will be flagged in the plot
        as only having partial data.
    xlab : str
        The x label to use in the plot.
    ylab : str
        The y label to use in the plot.
    title : str
        The title of the plot.
    plt_root : str
        The directory to save plots to.
    """
    plt.scatter(x, dat[:, 0], alpha=0.5, label="x")
    plt.scatter(x, dat[:, 1], alpha=0.5, label="y")
    plt.scatter(x, dat[:, 2], alpha=0.5, label="z")
    plt.scatter(x[missing], dat[missing, 0], color="gray", marker="1")
    plt.scatter(x[missing], dat[missing, 1], color="gray", marker="1")
    plt.scatter(x[missing], dat[missing, 2], color="gray", marker="1")
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(
        os.path.join(plt_root, f"{title.lower().replace(' ' , '_')}.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_all_dir(x: NDArray[np.float64], dat: NDArray[np.float64], direction: NDArray[np.float64], missing: list[int], xlab: str, ylab: str, title: str, plt_root: str):
    """
    Plot data with directions as seperate lines on the same plot. 
    The plot will be saved to `plt_root/{title.lower().replace(' ' , '_')}.png`.

    Parameters
    ----------
    x : NDArray[np.float64]
        The x data to plot against.
        Should have shape `(npoint,)`.
    dat : NDArray[np.float64]
        The data to plot.
        Should have shape `(npoint,)`.
    direction : NDArray[np.float64]
        The direction of motion at each data point.
        Should have shape `(npoint,)`.
    missing : list[int]
        A list of indices that will be flagged in the plot
        as only having partial data.
    xlab : str
        The x label to use in the plot.
    ylab : str
        The y label to use in the plot.
    title : str
        The title of the plot.
    plt_root : str
        The directory to save plots to.
    """
    plt.scatter(
        x[direction == 0],
        dat[direction == 0],
        color="black",
        alpha=0.5,
        label="Stationary",
    )
    plt.scatter(
        x[direction < 0],
        dat[direction < 0],
        color="blue",
        alpha=0.5,
        label="Decreasing",
    )
    plt.scatter(
        x[direction > 0],
        dat[direction > 0],
        color="red",
        alpha=0.5,
        label="Increasing",
    )
    plt.scatter(x[missing], dat[missing], color="gray", alpha=1, marker="1")
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(
        os.path.join(plt_root, f"{title.lower().replace(' ' , '_')}.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_hist(dat: NDArray[np.float64], direction: NDArray[np.float64], xlab: str, title: str, plt_root: str):
    """
    Plot a histogram of data with directions as seperate histograms on the same plot. 
    The plot will be saved to `plt_root/{title.lower().replace(' ' , '_')}.png`.

    Parameters
    ----------
    dat : NDArray[np.float64]
        The data to plot.
        Should have shape `(npoint,)`.
    direction : NDArray[np.float64]
        The direction of motion at each data point.
        Should have shape `(npoint,)`.
    xlab : str
        The x label to use in the plot.
    title : str
        The title of the plot.
    plt_root : str
        The directory to save plots to.
    """
    if len(direction == 0) > 0 and np.sum(np.isfinite(dat[direction == 0])) > 0:
        plt.hist(
            dat[direction == 0],
            bins="auto",
            color="black",
            alpha=0.5,
            label="Stationary",
        )
    if len(direction < 0) > 0 and np.sum(np.isfinite(dat[direction < 0])) > 0:
        plt.hist(
            dat[direction < 0],
            bins="auto",
            color="blue",
            alpha=0.5,
            label="Decreasing",
        )
    if len(direction > 0) > 0 and np.sum(np.isfinite(dat[direction > 0])) > 0:
        plt.hist(
            dat[direction > 0],
            bins="auto",
            color="red",
            alpha=0.5,
            label="Increasing",
        )
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel("Counts (#)")
    plt.title(title)
    plt.savefig(
        os.path.join(plt_root, f"{title.lower().replace(' ' , '_')}.png"),
        bbox_inches="tight",
    )
    plt.close()
