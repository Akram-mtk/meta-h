"""
Plotting utilities for the TP - MÉTA interface
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def surface_plot(fn, lb, ub, title="Function", n_points=80):
    """Create a 3D surface plot of a 2D view of the function."""
    x = np.linspace(lb, ub, n_points)
    y = np.linspace(lb, ub, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="none", alpha=0.9)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def contour_scatter(fn, lb, ub, population=None, best=None,
                    trail=None, title="Search History", n_points=80):
    """Contour plot with the population scattered on it."""
    x = np.linspace(lb, ub, n_points)
    y = np.linspace(lb, ub, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fn(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.contour(X, Y, Z, levels=15, cmap=cm.viridis)

    # Trail: path of all positions over iterations (small dots)
    if trail is not None and len(trail) > 0:
        for pos in trail:
            ax.scatter(pos[:, 0], pos[:, 1], c="black", s=5, alpha=0.3)

    # Current population
    if population is not None:
        ax.scatter(population[:, 0], population[:, 1],
                   c="orange", s=30, edgecolors="black", linewidths=0.5,
                   label="Population")

    # Best solution
    if best is not None:
        ax.scatter(best[0], best[1], c="red", s=100,
                   edgecolors="black", linewidths=1.0, zorder=5,
                   label="Best")

    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def line_plot(values, title="", xlabel="Iteration", ylabel="Fitness", color="red"):
    """Simple line plot for convergence / trajectory / average fitness."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(np.arange(len(values)), values, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
