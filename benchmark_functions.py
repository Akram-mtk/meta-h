"""
Benchmark Functions Module
Defines standard continuous optimization benchmark problems used in metaheuristics research.
These functions are used to test and evaluate optimization algorithms.
"""

import numpy as np


# ============================================================
#  BENCHMARK FUNCTIONS
# ============================================================

def F1(x):
    """Sphere function - simple, unimodal, convex"""
    return np.sum(x**2)


def F2(x):
    """Schwefel's problem 221 - sum of absolute values plus product"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def F5(x):
    """Rosenbrock function - classic unimodal problem"""
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def F7(x):
    """Sum of power function with random noise"""
    n = len(x)
    return np.sum((np.arange(1, n+1)) * x**4) + np.random.random()


def F8(x):
    """Schwefel's sine function - multimodal"""
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))


def F9(x):
    """Rastrigin function - highly multimodal, many local optima"""
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)


def F11(x):
    """Penalized function - product of cosines"""
    n = len(x)
    i = np.arange(1, n+1)
    return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(i)))


# Metadata for all benchmark functions
FUNCTIONS = {
    "F1-UM": {
        "func": F1,
        "lb": -100,
        "ub": 100,
        "latex": r"f(x) = \sum_{i=1}^{D} x_i^2"
    },
    "F2-UM": {
        "func": F2,
        "lb": -10,
        "ub": 10,
        "latex": r"f(x) = \sum_{i=1}^{D} |x_i| + \prod_{i=1}^{D} |x_i|"
    },
    "F5-UM": {
        "func": F5,
        "lb": -30,
        "ub": 30,
        "latex": r"f(x) = \sum_{i=1}^{D-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]"
    },
    "F7-UM": {
        "func": F7,
        "lb": -128,
        "ub": 128,
        "latex": r"f(x) = \sum_{i=1}^{D} i \cdot x_i^4 + \text{random}(0,1)"
    },
    "F8-MM": {
        "func": F8,
        "lb": -500,
        "ub": 500,
        "latex": r"f(x) = \sum_{i=1}^{D} -x_i \sin\left(\sqrt{|x_i|}\right)"
    },
    "F9-MM": {
        "func": F9,
        "lb": -5.12,
        "ub": 5.12,
        "latex": r"f(x) = \sum_{i=1}^{D} \left[x_i^2 - 10\cos(2\pi x_i) + 10\right]"
    },
    "F11-MM": {
        "func": F11,
        "lb": -600,
        "ub": 600,
        "latex": r"f(x) = 1 + \frac{1}{4000}\sum_{i=1}^{D} x_i^2 - \prod_{i=1}^{D} \cos\left(\frac{x_i}{\sqrt{i}}\right)"
    },
}


def evaluate_population(pop, fitness_func):
    """
    Evaluate all solutions in a population.
    
    Parameters:
    -----------
    pop : ndarray of shape (n_solutions, n_dimensions)
        The population to evaluate
    fitness_func : callable
        The fitness function to use
        
    Returns:
    --------
    fitness : ndarray of shape (n_solutions,)
        Fitness values for each solution
    best_idx : int
        Index of the best (minimum fitness) solution
    worst_idx : int
        Index of the worst (maximum fitness) solution
    """
    fitness = np.array([fitness_func(sol) for sol in pop])
    best_idx = np.argmin(fitness)
    worst_idx = np.argmax(fitness)
    return fitness, best_idx, worst_idx
