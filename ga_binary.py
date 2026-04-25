"""
Binary Genetic Algorithm (TP N°7 & TP N°8)
Supports selection / crossover / replacement variants specified in the PDFs.
MÉTA - Master 2 SII - USTHB
"""
import numpy as np


# =============================================================================
# Selection operators
# =============================================================================
def _select_random(pop, fitness, rng):
    """Random selection (TP7 Part 1)."""
    i = rng.integers(0, len(pop))
    j = rng.integers(0, len(pop))
    return pop[i].copy(), pop[j].copy()


def _select_roulette(pop, fitness, rng):
    """Roulette-wheel selection (TP8 variant 1).

    We are MINIMIZING fitness, so transform it:
        weight_i = (max - f_i) + eps
    Larger weights → more likely to be picked.
    """
    f = np.asarray(fitness, dtype=float)
    w = (f.max() - f) + 1e-12
    s = w.sum()
    cum = np.cumsum(w / s)

    r1, r2 = rng.random(), rng.random()
    i1 = int(np.searchsorted(cum, r1))
    i2 = int(np.searchsorted(cum, r2))
    i1 = min(i1, len(pop) - 1)
    i2 = min(i2, len(pop) - 1)
    return pop[i1].copy(), pop[i2].copy()


SELECTION_OPS = {
    "Random": _select_random,
    "Roulette": _select_roulette,
}


# =============================================================================
# Crossover operators
# =============================================================================
def _crossover_1point(p1, p2, rng):
    """1-point crossover (TP7 Part 1)."""
    D = len(p1)
    k = rng.integers(1, D)
    c1 = np.concatenate([p1[:k], p2[k:]])
    c2 = np.concatenate([p2[:k], p1[k:]])
    return c1, c2


def _crossover_2point(p1, p2, rng):
    """2-point crossover (TP8 variant 2)."""
    D = len(p1)
    k1, k2 = sorted(rng.choice(np.arange(1, D), size=2, replace=False))
    c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:]])
    c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:]])
    return c1, c2


def _crossover_3point(p1, p2, rng):
    """3-point crossover. Three cut points split each parent into 4 segments;
    children alternate segments between the two parents."""
    D = len(p1)
    if D < 4:
        # Fall back to 1-point for very short solutions
        return _crossover_1point(p1, p2, rng)
    k1, k2, k3 = sorted(rng.choice(np.arange(1, D), size=3, replace=False))
    c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:k3], p2[k3:]])
    c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:k3], p1[k3:]])
    return c1, c2


CROSSOVER_OPS = {
    "1-Point": _crossover_1point,
    "2-Point": _crossover_2point,
    "3-Point": _crossover_3point,
}


# =============================================================================
# Replacement operators
# =============================================================================
def _replace_children(P, fP, newP, fNew, rng):
    """Generational: P ← P_new (TP7 Part 1)."""
    return newP, fNew


def _replace_elitist(P, fP, newP, fNew, rng):
    """Elitist replacement: keep the N best from union(P, P_new) (TP8 variant 3)."""
    n = len(P)
    union = np.vstack([P, newP])
    fit = np.concatenate([fP, fNew])
    order = np.argsort(fit)[:n]
    return union[order], fit[order]


REPLACEMENT_OPS = {
    "Children": _replace_children,
    "Elitist": _replace_elitist,
}


# =============================================================================
# Main GA driver
# =============================================================================
def ga_binary(fitness_fn, dim,
              pop_size=30, max_iter=200,
              pc=0.7, pm=0.1,
              selection="Random",
              crossover="1-Point",
              replacement="Children",
              initial_population=None,
              seed=None):
    """
    Binary Genetic Algorithm for feature selection.

    Parameters
    ----------
    fitness_fn : callable taking a {0,1}-vector → float to MINIMIZE
    dim        : length of the solution vector (number of features)
    pop_size   : population size N
    max_iter   : number of generations T
    pc, pm     : crossover / mutation probabilities
    selection, crossover, replacement : operator names (see dicts above)
    """
    rng = np.random.default_rng(seed)
    sel_op = SELECTION_OPS[selection]
    cx_op = CROSSOVER_OPS[crossover]
    rep_op = REPLACEMENT_OPS[replacement]

    # --- Init ---
    if initial_population is not None:
        P = (np.asarray(initial_population) > 0.5).astype(int)
        pop_size, dim = P.shape
    else:
        P = rng.integers(0, 2, size=(pop_size, dim))

    fP = np.array([fitness_fn(x) for x in P])

    best_idx = int(np.argmin(fP))
    g = P[best_idx].copy()
    g_fit = float(fP[best_idx])

    convergence = [g_fit]
    avg_curve = [float(np.mean(fP))]
    trajectory = [float(P[0, 0])]

    for t in range(1, max_iter + 1):
        newP = []
        # Produce pop_size children (pairs)
        for _ in range(pop_size // 2):
            p1, p2 = sel_op(P, fP, rng)

            # Crossover
            if rng.random() < pc:
                c1, c2 = cx_op(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Bit-flip mutation
            for child in (c1, c2):
                mask = rng.random(dim) < pm
                child[mask] = 1 - child[mask]

            newP.append(c1)
            newP.append(c2)

        newP = np.array(newP, dtype=int)
        fNew = np.array([fitness_fn(x) for x in newP])

        # Replacement
        P, fP = rep_op(P, fP, newP, fNew, rng)

        # Update global best
        cur_best = int(np.argmin(fP))
        if fP[cur_best] < g_fit:
            g_fit = float(fP[cur_best])
            g = P[cur_best].copy()

        convergence.append(g_fit)
        avg_curve.append(float(np.mean(fP)))
        trajectory.append(float(P[0, 0]))

    return {
        "best_solution": g,
        "best_fitness": g_fit,
        "convergence_curve": np.array(convergence),
        "average_fitness_curve": np.array(avg_curve),
        "trajectory_first": np.array(trajectory),
    }
