"""
Particle Swarm Optimization (PSO)
TP - MÉTA - Master 2 SII - USTHB
"""
import numpy as np


def pso(fn, dim, lb, ub,
        pop_size=30, max_iter=200,
        w=0.3, c1=1.4, c2=1.4,
        initial_population=None,
        stagnation_tol=3,
        seed=None):
    """
    Particle Swarm Optimization.

    Returns a dict with:
        best_solution          - best solution found (vector of size dim)
        best_fitness           - best fitness value
        convergence_curve      - best fitness per iteration
        average_fitness_curve  - mean fitness per iteration
        trajectory_first       - value of x_1^(1) of the first particle per iteration
        position_history       - list (length = iterations+1) of positions (pop_size x dim)
        initial_population     - the initial population used
        initial_best           - best fitness in initial population
        initial_worst          - worst fitness in initial population
        stagnation_iter        - iteration at which stagnation was detected, or None
    """
    rng = np.random.default_rng(seed)

    # --- Initialization ---
    if initial_population is not None:
        X = np.array(initial_population, dtype=float)
        pop_size, dim = X.shape
    else:
        X = rng.uniform(lb, ub, size=(pop_size, dim))

    V = np.zeros((pop_size, dim))
    Vmax = 0.2 * (ub - lb)

    # Evaluate initial fitness
    fitness = np.array([fn(x) for x in X])
    initial_best = float(np.min(fitness))
    initial_worst = float(np.max(fitness))

    # Personal bests
    P = X.copy()
    P_fitness = fitness.copy()

    # Global best
    best_idx = int(np.argmin(fitness))
    g = X[best_idx].copy()
    g_fit = float(fitness[best_idx])

    # --- Tracking structures ---
    convergence_curve = [g_fit]
    average_fitness_curve = [float(np.mean(fitness))]
    trajectory_first = [float(X[0, 0])]
    position_history = [X.copy()]

    prev_g_fit = g_fit
    stagnation_count = 0
    stagnation_iter = None

    # --- Main loop ---
    for t in range(1, max_iter + 1):
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))

        # Velocity & position update (only if global best differs from particle)
        V = w * V + c1 * r1 * (g - X) + c2 * r2 * (P - X)
        V = np.clip(V, -Vmax, Vmax)
        X = X + V
        X = np.clip(X, lb, ub)

        # Evaluate
        fitness = np.array([fn(x) for x in X])

        # Update personal bests
        improved = fitness < P_fitness
        P[improved] = X[improved]
        P_fitness[improved] = fitness[improved]

        # Update global best
        best_idx = int(np.argmin(P_fitness))
        if P_fitness[best_idx] < g_fit:
            g_fit = float(P_fitness[best_idx])
            g = P[best_idx].copy()

        convergence_curve.append(g_fit)
        average_fitness_curve.append(float(np.mean(fitness)))
        trajectory_first.append(float(X[0, 0]))
        position_history.append(X.copy())

        # Stagnation detection (best did not change)
        if abs(g_fit - prev_g_fit) < 1e-12:
            stagnation_count += 1
            if stagnation_count >= stagnation_tol and stagnation_iter is None:
                stagnation_iter = t
        else:
            stagnation_count = 0
        prev_g_fit = g_fit

    return {
        "best_solution": g,
        "best_fitness": g_fit,
        "convergence_curve": np.array(convergence_curve),
        "average_fitness_curve": np.array(average_fitness_curve),
        "trajectory_first": np.array(trajectory_first),
        "position_history": position_history,
        "initial_population": position_history[0],
        "initial_best": initial_best,
        "initial_worst": initial_worst,
        "stagnation_iter": stagnation_iter,
    }
