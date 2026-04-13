import numpy as np

# F1 — Sphere function
def F1(x):
    return np.sum(x**2)

# F2 — Sum + Product of abs
def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# F5 — Rosenbrock
def F5(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# F7 — Quartic with noise
def F7(x):
    n = len(x)
    return np.sum((np.arange(1, n+1)) * x**4) + np.random.random()

# F8 — Schwefel
def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

# F9 — Rastrigin
def F9(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

# F11 — Griewank
def F11(x):
    n = len(x)
    i = np.arange(1, n+1)
    return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(i)))




def evaluate_population(pop, fitness_func):
    """Evaluate all solutions, return fitness values, best and worst"""
    fitness = np.array([fitness_func(sol) for sol in pop])
    best_idx = np.argmin(fitness)   # minimization
    worst_idx = np.argmax(fitness)
    return fitness, best_idx, worst_idx