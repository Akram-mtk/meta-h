"""
Genetic Algorithm (GA) - 3rd Variant (continuous)
TP - MÉTA - Master 2 SII - USTHB
"""
import numpy as np


def ga(fn, dim, lb, ub,
       pop_size=30, max_iter=200,
       pc=0.8, pm=0.1,
       initial_population=None,
       seed=None):
    """
    Simple real-coded Genetic Algorithm with:
        - tournament selection
        - single-point arithmetic crossover
        - uniform random mutation per gene
        - elitism (keep N best from union of P and P_new)
    """
    rng = np.random.default_rng(seed)

    if initial_population is not None:
        X = np.array(initial_population, dtype=float)
        pop_size, dim = X.shape
    else:
        X = rng.uniform(lb, ub, size=(pop_size, dim))

    fitness = np.array([fn(x) for x in X])
    initial_best = float(np.min(fitness))
    initial_worst = float(np.max(fitness))

    best_idx = int(np.argmin(fitness))
    g = X[best_idx].copy()
    g_fit = float(fitness[best_idx])

    convergence_curve = [g_fit]
    average_fitness_curve = [float(np.mean(fitness))]
    trajectory_first = [float(X[0, 0])]
    position_history = [X.copy()]

    for t in range(1, max_iter + 1):
        # --- Selection (tournament of size 2) ---
        new_pop = []
        for _ in range(pop_size // 2):
            # Select two parents via tournament
            a, b = rng.integers(0, pop_size, 2)
            p1 = X[a] if fitness[a] < fitness[b] else X[b]
            a, b = rng.integers(0, pop_size, 2)
            p2 = X[a] if fitness[a] < fitness[b] else X[b]

            # --- Crossover (arithmetic) ---
            if rng.random() < pc:
                alpha = rng.random(dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()

            # --- Mutation ---
            for child in (c1, c2):
                mask = rng.random(dim) < pm
                child[mask] = rng.uniform(lb, ub, mask.sum())

            new_pop.append(c1)
            new_pop.append(c2)

        new_pop = np.array(new_pop)
        new_pop = np.clip(new_pop, lb, ub)
        new_fit = np.array([fn(x) for x in new_pop])

        # --- Replacement (elitism from union) ---
        union = np.vstack([X, new_pop])
        union_fit = np.concatenate([fitness, new_fit])
        order = np.argsort(union_fit)[:pop_size]
        X = union[order]
        fitness = union_fit[order]

        # Update global best
        if fitness[0] < g_fit:
            g_fit = float(fitness[0])
            g = X[0].copy()

        convergence_curve.append(g_fit)
        average_fitness_curve.append(float(np.mean(fitness)))
        trajectory_first.append(float(X[0, 0]))
        position_history.append(X.copy())

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
    }
