"""
TP8 - Genetic Algorithm Variants Module
Implements three variants of GA with different crossover and replacement strategies:
  - Variant 1: One-point crossover, children-only replacement
  - Variant 2: Two-point crossover, children-only replacement
  - Variant 3: Multi-point crossover (3-point), best selection replacement
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
#  VARIANT 1: ONE-POINT CROSSOVER + CHILDREN REPLACEMENT
# ============================================================

def GA_variant1(X_train, X_test, y_train, y_test,
                N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """
    Genetic Algorithm Variant 1: One-point crossover with children replacement.
    
    Uses:
    - Selection: Random selection of two parents
    - Crossover: One-point crossover (split at random position)
    - Mutation: Bit-flip mutation
    - Replacement: Children-only (full generational replacement)
    
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
        
        # Create offspring using one-point crossover
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
        
        # Children-only replacement (full generational replacement)
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


# ============================================================
#  VARIANT 2: TWO-POINT CROSSOVER + CHILDREN REPLACEMENT
# ============================================================

def GA_variant2(X_train, X_test, y_train, y_test,
                N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """
    Genetic Algorithm Variant 2: Two-point crossover with children replacement.
    
    Uses:
    - Selection: Random selection of two parents
    - Crossover: Two-point crossover (split at two random positions)
    - Mutation: Bit-flip mutation
    - Replacement: Children-only (full generational replacement)
    
    Two-point crossover exchanges the middle segment between two crossover points,
    which often preserves more genetic structure than one-point crossover.
    
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
        
        # Create offspring using two-point crossover
        for _ in range(N // 2):
            # Random parent selection
            p1 = pop[np.random.randint(N)].copy()
            p2 = pop[np.random.randint(N)].copy()
            
            # Two-point crossover
            if np.random.random() < Rc:
                # Choose two random crossover points
                k1 = np.random.randint(1, D)
                k2 = np.random.randint(k1 + 1, D)
                
                # Exchange the middle segment
                c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:]])
                c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Bit-flip mutation
            for j in range(D):
                if np.random.random() < Rm:
                    c1[j] = 1 - c1[j]
                if np.random.random() < Rm:
                    c2[j] = 1 - c2[j]
            
            new_pop.extend([c1, c2])
        
        # Children-only replacement (full generational replacement)
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


# ============================================================
#  VARIANT 3: THREE-POINT CROSSOVER + BEST SELECTION REPLACEMENT
# ============================================================

def GA_variant3(X_train, X_test, y_train, y_test,
                N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """
    Genetic Algorithm Variant 3: Three-point crossover with best selection replacement.
    
    Uses:
    - Selection: Random selection of two parents
    - Crossover: Three-point crossover (split at three random positions)
    - Mutation: Bit-flip mutation
    - Replacement: Best selection (keep best from union of parents and offspring)
    
    Three-point crossover divides the chromosome into 4 segments and exchanges segments.
    Best selection replacement is more elitist - keeps the best solutions from both
    old and new populations, which can improve convergence but reduce diversity.
    
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
        offspring = []
        offspring_fit = []
        
        # Create offspring using three-point crossover
        for _ in range(N // 2):
            # Random parent selection
            p1 = pop[np.random.randint(N)].copy()
            p2 = pop[np.random.randint(N)].copy()
            
            # Three-point crossover
            if np.random.random() < Rc:
                # Choose three random crossover points
                points = np.sort(np.random.choice(range(1, D), 3, replace=False))
                k1, k2, k3 = points
                
                # Exchange segments: p1[k1:k2] with p2[k1:k2], p1[k3:] with p2[k3:]
                c1 = np.concatenate([p1[:k1], p2[k1:k2], p1[k2:k3], p2[k3:]])
                c2 = np.concatenate([p2[:k1], p1[k1:k2], p2[k2:k3], p1[k3:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Bit-flip mutation
            for j in range(D):
                if np.random.random() < Rm:
                    c1[j] = 1 - c1[j]
                if np.random.random() < Rm:
                    c2[j] = 1 - c2[j]
            
            offspring.append(c1)
            offspring.append(c2)
        
        # Evaluate offspring
        offspring = np.array(offspring[:N])
        for i in range(len(offspring)):
            sel = np.where(offspring[i] == 1)[0]
            if len(sel) == 0:
                offspring_fit.append(1.0)
                continue
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train[:, sel], y_train)
            acc = accuracy_score(y_test, knn.predict(X_test[:, sel]))
            fit = alpha*(1-acc) + (1-alpha)*(len(sel)/D)
            offspring_fit.append(fit)
        
        # Best selection replacement: Keep N best from union of pop and offspring
        combined_pop = np.vstack([pop, offspring[:len(offspring_fit)]])
        combined_fit = np.concatenate([fit_vals, offspring_fit])
        
        # Sort by fitness and select top N
        sorted_indices = np.argsort(combined_fit)[:N]
        pop = combined_pop[sorted_indices]
        fit_vals = combined_fit[sorted_indices]
        
        # Update global best
        old = gBest_fit
        for i in range(N):
            if fit_vals[i] < gBest_fit:
                sel = np.where(pop[i] == 1)[0]
                gBest = pop[i].copy()
                gBest_fit = fit_vals[i]
                if len(sel) > 0:
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train[:, sel], y_train)
                    gBest_acc = accuracy_score(y_test, knn.predict(X_test[:, sel]))
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
