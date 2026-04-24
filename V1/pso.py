"""
Particle Swarm Optimization (PSO) Module
Implements the PSO algorithm for continuous and feature selection optimization problems.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
#  PSO FOR CONTINUOUS OPTIMIZATION
# ============================================================

def PSO(fitness_func, n_particles, dim, lb, ub,
        w=0.5, c1=2, c2=2, max_iter=200, k=0.2):
    """
    Particle Swarm Optimization for continuous optimization problems.
    
    Parameters:
    -----------
    fitness_func : callable
        The objective function to minimize
    n_particles : int
        Number of particles in the swarm
    dim : int
        Dimensionality of the problem
    lb : float
        Lower bound for all dimensions
    ub : float
        Upper bound for all dimensions
    w : float, default=0.5
        Inertia weight controlling momentum
    c1 : float, default=2
        Cognitive parameter (attraction to personal best)
    c2 : float, default=2
        Social parameter (attraction to global best)
    max_iter : int, default=200
        Maximum number of iterations
    k : float, default=0.2
        Velocity clamping factor (as fraction of search range)
        
    Returns:
    --------
    dict
        Results including global best, convergence curve, and population statistics
    """
    vMax = k * (ub - lb)
    
    # Initialize positions and velocities
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))
    
    # Evaluate initial population
    fitness = np.array([fitness_func(x) for x in X])
    pBest = X.copy()
    pBest_fit = fitness.copy()
    
    # Track global best
    gBest_idx = np.argmin(fitness)
    gBest = X[gBest_idx].copy()
    gBest_fit = fitness[gBest_idx]
    
    # Statistics tracking
    init_best = gBest_fit
    init_worst = np.max(fitness)
    init_pop = X.copy()
    convergence = [gBest_fit]
    avg_fitness_history = [np.mean(fitness)]
    trajectory = [X[0, 0]]
    stagnation_counter = 0
    stagnation_iter = 0
    all_positions = [X.copy()]

    # Main loop
    for t in range(max_iter):
        for i in range(n_particles):
            # Skip updating the global best particle
            if not np.array_equal(X[i], gBest):
                r1, r2 = np.random.random(dim), np.random.random(dim)
                
                # Update velocity with PSO equation
                V[i] = w*V[i] + c1*r1*(pBest[i]-X[i]) + c2*r2*(gBest-X[i])
                V[i] = np.clip(V[i], -vMax, vMax)
                
                # Update position
                X[i] = np.clip(X[i] + V[i], lb, ub)
        
        # Evaluate new population
        fitness = np.array([fitness_func(x) for x in X])
        
        # Update personal bests
        improved = fitness < pBest_fit
        pBest[improved] = X[improved].copy()
        pBest_fit[improved] = fitness[improved]
        
        # Update global best
        best_idx = np.argmin(fitness)
        old_gBest_fit = gBest_fit
        if fitness[best_idx] < gBest_fit:
            gBest = X[best_idx].copy()
            gBest_fit = fitness[best_idx]
            stagnation_iter = t + 1
        
        # Track metrics
        convergence.append(gBest_fit)
        avg_fitness_history.append(np.mean(fitness))
        trajectory.append(X[0, 0])
        all_positions.append(X.copy())
        
        # Check for stagnation (early stopping)
        if gBest_fit == old_gBest_fit:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        if stagnation_counter >= 3:
            break

    return {
        "gBest": gBest,
        "gBest_fit": gBest_fit,
        "convergence": convergence,
        "avg_fitness": avg_fitness_history,
        "trajectory": trajectory,
        "stagnation_iter": stagnation_iter,
        "init_best": init_best,
        "init_worst": init_worst,
        "init_pop": init_pop,
        "final_pop": X.copy(),
        "all_positions": all_positions
    }


def run_multiple_PSO(fitness_func, runs, n_particles, dim, lb, ub,
                     w=0.5, c1=2, c2=2, max_iter=200):
    """
    Execute PSO multiple times and aggregate statistics.
    
    Parameters:
    -----------
    fitness_func : callable
        The objective function to minimize
    runs : int
        Number of independent runs
    n_particles : int
        Number of particles in the swarm
    dim : int
        Dimensionality of the problem
    lb : float
        Lower bound for all dimensions
    ub : float
        Upper bound for all dimensions
    w, c1, c2 : float
        PSO hyperparameters
    max_iter : int
        Maximum number of iterations per run
        
    Returns:
    --------
    dict
        Aggregated statistics across all runs (best, worst, mean, std, etc.)
    """
    all_best = []
    all_conv = []
    all_avg = []
    all_traj = []
    all_final_pops = []
    
    for _ in range(runs):
        r = PSO(fitness_func, n_particles, dim, lb, ub, w, c1, c2, max_iter)
        all_best.append(r["gBest_fit"])
        all_conv.append(r["convergence"])
        all_avg.append(r["avg_fitness"])
        all_traj.append(r["trajectory"])
        all_final_pops.append(r["final_pop"])
    
    results = np.array(all_best)
    
    # Pad convergence curves to same length for averaging
    ml = max(len(c) for c in all_conv)
    pad = lambda lst: [l + [l[-1]]*(ml-len(l)) for l in lst]
    
    return {
        "Best": np.min(results),
        "Worst": np.max(results),
        "AVG": np.mean(results),
        "STD": np.std(results, ddof=1),
        "mean_conv": np.mean(pad(all_conv), axis=0),
        "mean_avg": np.mean(pad(all_avg), axis=0),
        "mean_traj": np.mean(pad(all_traj), axis=0),
        "all_final_pops": all_final_pops
    }


# ============================================================
#  PSO FOR FEATURE SELECTION
# ============================================================

def fitness_fs_threshold(solution, X_train, X_test, y_train, y_test, alpha=0.9):
    """
    Fitness function for feature selection using threshold.
    Features with values > 0.5 are selected.
    
    Parameters:
    -----------
    solution : ndarray
        Particle position (feature weights in [0, 1])
    X_train, X_test : ndarray
        Training and test feature matrices
    y_train, y_test : ndarray
        Training and test labels
    alpha : float, default=0.9
        Balance between accuracy (alpha) and feature count (1-alpha)
        
    Returns:
    --------
    fit : float
        Fitness value to minimize
    acc : float
        Classification accuracy achieved
    selected : list
        Indices of selected features
    """
    D = len(solution)
    selected = np.where(np.array(solution) > 0.5)[0]
    
    # If no features selected, return worst fitness
    if len(selected) == 0:
        return 1.0, 0.0, []
    
    # Train KNN on selected features
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train[:, selected], y_train)
    acc = accuracy_score(y_test, knn.predict(X_test[:, selected]))
    
    # Fitness combines accuracy and feature reduction
    fit = alpha*(1-acc) + (1-alpha)*(len(selected)/D)
    return fit, acc, sorted(list(selected))


def PSO_feature_selection(X_train, X_test, y_train, y_test,
                          n_particles=10, max_iter=20, w=0.5, c1=2, c2=2,
                          alpha=0.99, k=0.2):
    """
    PSO applied to feature selection problem.
    
    Parameters:
    -----------
    X_train, X_test : ndarray
        Training and test feature matrices
    y_train, y_test : ndarray
        Training and test labels
    n_particles : int, default=10
        Number of particles
    max_iter : int, default=20
        Maximum iterations
    w, c1, c2 : float
        PSO hyperparameters
    alpha : float, default=0.99
        Balance between accuracy and feature reduction
    k : float, default=0.2
        Velocity clamping factor
        
    Returns:
    --------
    dict
        Results including best selected features, accuracy, and convergence
    """
    dim = X_train.shape[1]
    lb, ub = 0, 1
    vMax = k * (ub - lb)
    
    # Initialize swarm
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))
    pBest = X.copy()
    pBest_fit = np.full(n_particles, np.inf)
    gBest = None
    gBest_fit = np.inf
    gBest_acc = 0
    gBest_sel = []

    # Evaluate initial population
    for i in range(n_particles):
        fit, acc, sel = fitness_fs_threshold(X[i], X_train, X_test, y_train, y_test, alpha)
        pBest_fit[i] = fit
        if fit < gBest_fit:
            gBest = X[i].copy()
            gBest_fit = fit
            gBest_acc = acc
            gBest_sel = sel

    # Track metrics
    convergence = [gBest_fit]
    acc_history = [gBest_acc]
    avg_fitness = [np.mean(pBest_fit)]
    trajectory = [X[0, 0]]
    stag = 0
    all_positions = [X.copy()]

    # Main loop
    for t in range(max_iter):
        # Update positions
        for i in range(n_particles):
            if gBest is not None and not np.array_equal(X[i], gBest):
                r1, r2 = np.random.random(dim), np.random.random(dim)
                V[i] = w*V[i] + c1*r1*(pBest[i]-X[i]) + c2*r2*(gBest-X[i])
                V[i] = np.clip(V[i], -vMax, vMax)
                X[i] = np.clip(X[i] + V[i], lb, ub)
        
        # Evaluate new population
        old = gBest_fit
        fit_vals = []
        for i in range(n_particles):
            fit, acc, sel = fitness_fs_threshold(X[i], X_train, X_test, y_train, y_test, alpha)
            fit_vals.append(fit)
            if fit < pBest_fit[i]:
                pBest[i] = X[i].copy()
                pBest_fit[i] = fit
            if fit < gBest_fit:
                gBest = X[i].copy()
                gBest_fit = fit
                gBest_acc = acc
                gBest_sel = sel
        
        # Track metrics
        convergence.append(gBest_fit)
        acc_history.append(gBest_acc)
        avg_fitness.append(np.mean(fit_vals))
        trajectory.append(X[0, 0])
        all_positions.append(X.copy())
        
        # Check for stagnation
        if gBest_fit == old:
            stag += 1
        else:
            stag = 0
        
        if stag >= 3:
            break

    return {
        "gBest_fit": gBest_fit,
        "gBest_acc": gBest_acc,
        "gBest_sel": gBest_sel,
        "n_selected": len(gBest_sel),
        "convergence": convergence,
        "acc_history": acc_history,
        "avg_fitness": avg_fitness,
        "trajectory": trajectory,
        "all_positions": all_positions,
        "final_pop": X.copy()
    }
