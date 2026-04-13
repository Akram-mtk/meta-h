import numpy as np
from functions import *

def PSO(fitness_func, n_particles, dim, lb, ub, 
        w=0.5, c1=2, c2=2, max_iter=200, k=0.2):
    """
    Particle Swarm Optimization
    Returns: best_solution, best_fitness, convergence_curve
    """
    vMax = k * (ub - lb)
    
    # 1. Initialize population randomly
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))  # velocities start at 0
    
    # 2. Evaluate & find initial best
    fitness = np.array([fitness_func(x) for x in X])
    pBest = X.copy()               # personal best positions
    pBest_fit = fitness.copy()     # personal best fitness
    gBest_idx = np.argmin(fitness)
    gBest = X[gBest_idx].copy()    # global best position
    gBest_fit = fitness[gBest_idx] # global best fitness
    
    convergence = [gBest_fit]
    stagnation = 0
    
    # 3. Main loop
    for t in range(max_iter):
        for i in range(n_particles):
            if not np.array_equal(X[i], gBest):
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)
                
                # Update velocity
                V[i] = (w * V[i] 
                        + c1 * r1 * (pBest[i] - X[i]) 
                        + c2 * r2 * (gBest - X[i]))
                
                # Clamp velocity
                V[i] = np.clip(V[i], -vMax, vMax)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Clamp position
                X[i] = np.clip(X[i], lb, ub)
        
        # Evaluate
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
        
        convergence.append(gBest_fit)
        
        # Check stagnation
        if gBest_fit == old_gBest_fit:
            stagnation += 1
        else:
            stagnation = 0
        if stagnation >= 3:
            break
    
    return gBest, gBest_fit, convergence

# Run PSO on F1
best_sol, best_fit, curve = PSO(F1, n_particles=30, dim=30, 
                                 lb=-100, ub=100)
print(f"Best fitness: {best_fit:.6f}")










def run_pso_experiments(fitness_func, runs=50, n_particles=30, dim=30,
                        lb=-100, ub=100, w=0.5, c1=2, c2=2, max_iter=200):
    all_best = []
    all_curves = []
    all_avg_fitness = []
    all_trajectories = []
    
    for r in range(runs):
        best_sol, best_fit, curve = PSO(
            fitness_func, n_particles, dim, lb, ub,
            w=w, c1=c1, c2=c2, max_iter=max_iter
        )
        all_best.append(best_fit)
        all_curves.append(curve)
    
    results = np.array(all_best)
    return {
        'Best':  np.min(results),
        'Worst': np.max(results),
        'AVG':   np.mean(results),
        'STD':   np.std(results, ddof=1),
        'curves': all_curves
    }

# Test with c1=c2=2
res1 = run_pso_experiments(F1, c1=2, c2=2)
# Test with c1=c2=1
res2 = run_pso_experiments(F1, c1=1, c2=1)