"""
Genetic Algorithm (GA) Module
Implements a simple genetic algorithm for feature selection problems.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
#  GENETIC ALGORITHM FOR FEATURE SELECTION
# ============================================================

def GA_feature_selection(X_train, X_test, y_train, y_test,
                         N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """
    Genetic Algorithm for feature selection.
    
    Uses binary representation where each bit indicates whether a feature is selected.
    Employs one-point crossover and bit-flip mutation.
    
    Parameters:
    -----------
    X_train, X_test : ndarray
        Training and test feature matrices
    y_train, y_test : ndarray
        Training and test labels
    N : int, default=10
        Population size
    T : int, default=20
        Number of generations
    Rc : float, default=0.70
        Crossover probability
    Rm : float, default=0.10
        Mutation probability per bit
    alpha : float, default=0.99
        Balance between accuracy (alpha) and feature reduction (1-alpha)
        
    Returns:
    --------
    dict
        Results including best features, accuracy, and convergence
    """
    D = X_train.shape[1]
    
    # Initialize population with random binary chromosomes
    pop = np.random.randint(0, 2, (N, D))
    
    # Track global best
    gBest = None
    gBest_fit = np.inf
    gBest_acc = 0
    gBest_sel = []
    gBest_sf = 0
    fit_vals = np.zeros(N)

    # Evaluate initial population
    for i in range(N):
        sel = np.where(pop[i] == 1)[0]
        if len(sel) == 0:
            fit_vals[i] = 1.0
            continue
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train[:, sel], y_train)
        acc = accuracy_score(y_test, knn.predict(X_test[:, sel]))
        fit = alpha*(1-acc) + (1-alpha)*(len(sel)/D)
        fit_vals[i] = fit
        
        if fit < gBest_fit:
            gBest = pop[i].copy()
            gBest_fit = fit
            gBest_acc = acc
            gBest_sel = sorted(list(sel))
            gBest_sf = len(sel)

    # Track convergence
    convergence = [gBest_fit]
    avg_fitness = [np.mean(fit_vals)]
    trajectory = [float(pop[0][0])]
    stag = 0

    # Main generation loop
    for t in range(T):
        new_pop = []
        
        # Create offspring through selection, crossover, and mutation
        for _ in range(N // 2):
            # Random parent selection
            p1 = pop[np.random.randint(N)].copy()
            p2 = pop[np.random.randint(N)].copy()
            
            # One-point crossover
            if np.random.random() < Rc:
                k = np.random.randint(1, D)
                c1 = np.concatenate([p1[:k], p2[k:]])
                c2 = np.concatenate([p2[:k], p1[k:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Bit-flip mutation
            for j in range(D):
                if np.random.random() < Rm:
                    c1[j] = 1 - c1[j]
                if np.random.random() < Rm:
                    c2[j] = 1 - c2[j]
            
            new_pop.extend([c1, c2])
        
        # Replace population (children-only replacement strategy)
        pop = np.array(new_pop[:N])
        
        # Evaluate new population
        old = gBest_fit
        for i in range(N):
            sel = np.where(pop[i] == 1)[0]
            if len(sel) == 0:
                fit_vals[i] = 1.0
                continue
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train[:, sel], y_train)
            acc = accuracy_score(y_test, knn.predict(X_test[:, sel]))
            fit = alpha*(1-acc) + (1-alpha)*(len(sel)/D)
            fit_vals[i] = fit
            
            if fit < gBest_fit:
                gBest = pop[i].copy()
                gBest_fit = fit
                gBest_acc = acc
                gBest_sel = sorted(list(sel))
                gBest_sf = len(sel)
        
        # Track convergence metrics
        convergence.append(gBest_fit)
        avg_fitness.append(np.mean(fit_vals))
        trajectory.append(float(pop[0][0]))
        
        # Check for stagnation (early stopping)
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
        "n_selected": gBest_sf,
        "convergence": convergence,
        "avg_fitness": avg_fitness,
        "trajectory": trajectory
    }
