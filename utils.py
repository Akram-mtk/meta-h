"""
Data Utilities Module
Provides helper functions for dataset loading, preprocessing, and common operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split


# ============================================================
#  DATA LOADING
# ============================================================

def get_dataset(dataset_name="Synthetic", test_size=0.2, random_state=42):
    """
    Load and prepare a dataset for machine learning.
    
    Parameters:
    -----------
    dataset_name : str, default="Synthetic"
        Which dataset to load: "Synthetic" or "Digits"
    test_size : float, default=0.2
        Fraction of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : ndarray
        Training and test features and labels
    """
    if dataset_name == "Digits":
        # Load sklearn's digits dataset (8x8 images of handwritten digits)
        X, y = load_digits(return_X_y=True)
    else:
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=3,
            random_state=random_state
        )
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


# ============================================================
#  VISUALIZATION HELPERS
# ============================================================

def compute_grid(func, lb, ub, pts=100):
    """
    Compute function values on a 2D grid for visualization.
    
    Useful for plotting 2D benchmark functions and the search space.
    
    Parameters:
    -----------
    func : callable
        The function to evaluate on the grid
    lb : float
        Lower bound for grid range
    ub : float
        Upper bound for grid range
    pts : int, default=100
        Number of points per dimension (grid is pts × pts)
        
    Returns:
    --------
    Xg, Yg, Zg : ndarray
        Coordinate meshgrids and function values
    """
    x_r = np.linspace(lb, ub, pts)
    y_r = np.linspace(lb, ub, pts)
    Xg, Yg = np.meshgrid(x_r, y_r)
    Zg = np.zeros_like(Xg)
    
    for i in range(pts):
        for j in range(pts):
            Zg[i, j] = func(np.array([Xg[i, j], Yg[i, j]]))
    
    return Xg, Yg, Zg


def create_3d_surface_plot(Xg, Yg, Zg, title="3D Surface"):
    """
    Create a 3D surface plot of a function.
    
    Parameters:
    -----------
    Xg, Yg, Zg : ndarray
        Coordinate meshgrids from compute_grid()
    title : str
        Title for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.7)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    return fig


def create_contour_plot(Xg, Yg, Zg, pop=None, best_idx=None, title="Contour Plot"):
    """
    Create a contour plot of a function with optional population visualization.
    
    Parameters:
    -----------
    Xg, Yg, Zg : ndarray
        Coordinate meshgrids from compute_grid()
    pop : ndarray, optional
        Population solutions to plot (shape: n_solutions × 2)
    best_idx : int, optional
        Index of the best solution to highlight
    title : str
        Title for the plot
        
    Returns:
    --------
    fig, ax : tuple
        The created figure and axes
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(Xg, Yg, Zg, levels=30, cmap="viridis")
    
    if pop is not None:
        ax.scatter(pop[:, 0], pop[:, 1], c="black", s=15, label="Solution")
        if best_idx is not None:
            ax.scatter(pop[best_idx, 0], pop[best_idx, 1], 
                      c="red", s=100, label="Best solution", zorder=5)
    
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    if pop is not None:
        ax.legend()
    
    return fig, ax


def create_convergence_plot(convergence, title="Convergence Curve", color="red"):
    """
    Create a convergence plot showing best fitness over iterations.
    
    Parameters:
    -----------
    convergence : list or ndarray
        Best fitness value at each iteration
    title : str
        Title for the plot
    color : str
        Color for the line
        
    Returns:
    --------
    fig, ax : tuple
        The created figure and axes
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(convergence, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.grid(True, alpha=0.3)
    return fig, ax


# ============================================================
#  STATISTICS HELPERS
# ============================================================

def pad_convergence_curves(curves):
    """
    Pad convergence curves to the same length.
    
    Useful for averaging curves from multiple runs when they terminate at different times.
    
    Parameters:
    -----------
    curves : list of lists/arrays
        Multiple convergence curves of potentially different lengths
        
    Returns:
    --------
    list
        Padded curves, all same length
    """
    max_len = max(len(c) for c in curves)
    return [list(c) + [c[-1]] * (max_len - len(c)) for c in curves]


def aggregate_results(results_list):
    """
    Aggregate results from multiple runs.
    
    Parameters:
    -----------
    results_list : list of float
        Best fitness values from each run
        
    Returns:
    --------
    dict
        Aggregated statistics (best, worst, mean, std)
    """
    results = np.array(results_list)
    return {
        "Best": np.min(results),
        "Worst": np.max(results),
        "AVG": np.mean(results),
        "STD": np.std(results, ddof=1)
    }
