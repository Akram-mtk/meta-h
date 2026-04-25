"""
Benchmark Functions for Metaheuristic Optimization
TP - MÉTA - Master 2 SII - USTHB
"""
import numpy as np


# --------------------------------------------------------------------------
# Unimodal benchmark functions (UM)
# --------------------------------------------------------------------------
def F1(x):
    """Sphere function: f(x) = sum(x_i^2), range [-100, 100], min = 0"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def F2(x):
    """Schwefel 2.22: f(x) = sum(|x_i|) + prod(|x_i|), range [-10, 10], min = 0"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def F5(x):
    """Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2),
    range [-30, 30], min = 0"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def F7(x):
    """Noisy quartic: f(x) = sum(i * x_i^4) + random[0,1),
    range [-1.28, 1.28] (scaled to [-128, 128] per TP), min = 0"""
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(np.sum(i * (x ** 4)) + np.random.random())


# --------------------------------------------------------------------------
# Multimodal benchmark functions (MM)
# --------------------------------------------------------------------------
def F8(x):
    """Schwefel: f(x) = sum(-x_i * sin(sqrt(|x_i|))),
    range [-500, 500], min = -418.9829 * D"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))


def F9(x):
    """Rastrigin: f(x) = sum(x_i^2 - 10*cos(2*pi*x_i) + 10),
    range [-5.12, 5.12], min = 0"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))


def F11(x):
    """Griewank: f(x) = 1/4000 * sum(x_i^2) - prod(cos(x_i/sqrt(i))) + 1,
    range [-600, 600], min = 0"""
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float((1.0 / 4000.0) * np.sum(x ** 2) - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
FUNCTIONS = {
    "F1-UM": {
        "fn": F1,
        "range": (-100.0, 100.0),
        "min": 0.0,
        "type": "Unimodal",
        "formula": r"f(x) = \sum_{i=1}^{D} x_i^2",
        "description": "Sphere function",
    },
    "F2-UM": {
        "fn": F2,
        "range": (-10.0, 10.0),
        "min": 0.0,
        "type": "Unimodal",
        "formula": r"f(x) = \sum_{i=1}^{D} |x_i| + \prod_{i=1}^{D} |x_i|",
        "description": "Schwefel 2.22 function",
    },
    "F5-UM": {
        "fn": F5,
        "range": (-30.0, 30.0),
        "min": 0.0,
        "type": "Unimodal",
        "formula": r"f(x) = \sum_{i=1}^{D-1} \left[100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right]",
        "description": "Rosenbrock function",
    },
    "F7-UM": {
        "fn": F7,
        "range": (-128.0, 128.0),
        "min": 0.0,
        "type": "Unimodal",
        "formula": r"f(x) = \sum_{i=1}^{D} i \cdot x_i^4 + \text{random}[0,1)",
        "description": "Noisy Quartic function",
    },
    "F8-MM": {
        "fn": F8,
        "range": (-500.0, 500.0),
        "min": None,   # depends on D: -418.9829 * D
        "type": "Multimodal",
        "formula": r"f(x) = \sum_{i=1}^{D} -x_i \sin\left(\sqrt{|x_i|}\right)",
        "description": "Schwefel function",
    },
    "F9-MM": {
        "fn": F9,
        "range": (-5.12, 5.12),
        "min": 0.0,
        "type": "Multimodal",
        "formula": r"f(x) = \sum_{i=1}^{D} \left[ x_i^2 - 10\cos(2\pi x_i) + 10 \right]",
        "description": "Rastrigin function",
    },
    "F11-MM": {
        "fn": F11,
        "range": (-600.0, 600.0),
        "min": 0.0,
        "type": "Multimodal",
        "formula": r"f(x) = \frac{1}{4000}\sum_{i=1}^{D} x_i^2 - \prod_{i=1}^{D} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1",
        "description": "Griewank function",
    },
}


def get_function(name):
    """Return (fn, lb, ub, info) for a given function name."""
    info = FUNCTIONS[name]
    lb, ub = info["range"]
    return info["fn"], lb, ub, info


def get_min(name, dim):
    """Return the theoretical minimum of a function for a given dimension."""
    info = FUNCTIONS[name]
    if info["min"] is not None:
        return info["min"]
    # F8 min depends on dimension
    if name == "F8-MM":
        return -418.9829 * dim
    return 0.0
