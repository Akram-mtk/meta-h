import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
#  TP1 — BENCHMARK FUNCTIONS
# ============================================================

def F1(x):
    """Sphere function — range [-100, 100], fmin = 0"""
    return np.sum(x**2)

def F2(x):
    """Sum + Product of abs — range [-10, 10], fmin = 0"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F5(x):
    """Rosenbrock — range [-30, 30], fmin = 0"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def F7(x):
    """Quartic with noise — range [-128, 128], fmin = 0"""
    n = len(x)
    return np.sum(np.arange(1, n + 1) * x**4) + np.random.random()

def F8(x):
    """Schwefel — range [-500, 500], fmin = -418.9829 * n"""
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    """Rastrigin — range [-5.12, 5.12], fmin = 0"""
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F11(x):
    """Griewank — range [-600, 600], fmin = 0"""
    n = len(x)
    i = np.arange(1, n + 1)
    return 1 + (1 / 4000) * np.sum(x**2) - np.prod(np.cos(x / np.sqrt(i)))


# Dictionary of all functions with their configs
FUNCTIONS = {
    "F1":  {"func": F1,  "lb": -100,   "ub": 100,   "dim": 30},
    "F2":  {"func": F2,  "lb": -10,    "ub": 10,    "dim": 30},
    "F5":  {"func": F5,  "lb": -30,    "ub": 30,    "dim": 30},
    "F7":  {"func": F7,  "lb": -128,   "ub": 128,   "dim": 30},
    "F8":  {"func": F8,  "lb": -500,   "ub": 500,   "dim": 30},
    "F9":  {"func": F9,  "lb": -5.12,  "ub": 5.12,  "dim": 30},
    "F11": {"func": F11, "lb": -600,   "ub": 600,   "dim": 30},
}


# ============================================================
#  TP2 — POPULATION INITIALIZATION & EVALUATION
# ============================================================

def generate_population(size, dim, lb, ub):
    """Generate a random population of solutions within bounds."""
    return np.random.uniform(lb, ub, (size, dim))


def load_population(filename):
    """Load a population from a semicolon-separated CSV file."""
    return np.loadtxt(filename, delimiter=";")


def evaluate_population(pop, fitness_func):
    """
    Evaluate all solutions in the population.
    Returns: fitness array, best index, worst index.
    """
    fitness = np.array([fitness_func(sol) for sol in pop])
    best_idx = np.argmin(fitness)
    worst_idx = np.argmax(fitness)
    return fitness, best_idx, worst_idx


def run_multiple_populations(fitness_func, runs, pop_size, dim, lb, ub):
    """
    TP2: Run multiple random populations and compute statistics.
    Returns: Best, Worst, AVG, STD.
    """
    best_per_run = []

    global_best = None
    global_worst = None

    for r in range(runs):
        pop = generate_population(pop_size, dim, lb, ub)
        fitness, best_i, worst_i = evaluate_population(pop, fitness_func)

        best_per_run.append(fitness[best_i])

        if global_best is None or fitness[best_i] < global_best:
            global_best = fitness[best_i]
        if global_worst is None or fitness[worst_i] > global_worst:
            global_worst = fitness[worst_i]

    best_per_run = np.array(best_per_run)
    return {
        "Best": np.min(best_per_run),
        "Worst": np.max(best_per_run),
        "AVG": np.mean(best_per_run),
        "STD": np.std(best_per_run, ddof=1),
    }


def plot_population_on_contour(pop, fitness_func, lb, ub, title="Population"):
    """Plot solutions on a 2D contour map (using x1, x2 only)."""
    x = np.linspace(lb, ub, 200)
    y = np.linspace(lb, ub, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_func(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30, cmap="viridis")
    plt.scatter(pop[:, 0], pop[:, 1], c="black", s=20, label="Solutions")

    # Mark best
    fitness_vals = [fitness_func(s) for s in pop]
    best_i = np.argmin(fitness_vals)
    plt.scatter(pop[best_i, 0], pop[best_i, 1], c="red", s=120, marker="*",
                label="Best", zorder=5)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  TP3 — PARTICLE SWARM OPTIMIZATION
# ============================================================

def PSO(fitness_func, n_particles, dim, lb, ub,
        w=0.5, c1=2, c2=2, max_iter=200, k=0.2):
    """
    Particle Swarm Optimization algorithm.

    Returns:
        gBest       — best solution found
        gBest_fit   — fitness of best solution
        convergence — list of best fitness at each iteration
        avg_fitness — list of average population fitness at each iteration
        trajectory  — list of x1 values of particle 0 at each iteration
    """
    vMax = k * (ub - lb)

    # 1. Initialize population and velocities
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))

    # 2. Evaluate initial population
    fitness = np.array([fitness_func(x) for x in X])

    pBest = X.copy()
    pBest_fit = fitness.copy()

    gBest_idx = np.argmin(fitness)
    gBest = X[gBest_idx].copy()
    gBest_fit = fitness[gBest_idx]

    convergence = [gBest_fit]
    avg_fitness_history = [np.mean(fitness)]
    trajectory = [X[0, 0]]  # track x1 of first particle
    stagnation_counter = 0
    stagnation_iter = 0

    # 3. Main loop
    for t in range(max_iter):
        for i in range(n_particles):
            # Only update if particle is not already the best
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

        # Evaluate new positions
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

        # Track history
        convergence.append(gBest_fit)
        avg_fitness_history.append(np.mean(fitness))
        trajectory.append(X[0, 0])

        # Check stagnation (stop if no improvement for 3 iterations)
        if gBest_fit == old_gBest_fit:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= 3:
            break

    return gBest, gBest_fit, convergence, avg_fitness_history, trajectory, stagnation_iter


def plot_pso_results(convergence, avg_fitness, trajectory, title="PSO"):
    """Plot the 3 charts required by the teacher for TP3."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Convergence curve
    axes[0].plot(convergence, color="red")
    axes[0].set_title("Convergence Curve")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best Fitness")

    # 2. Trajectory of first particle
    axes[1].plot(trajectory, color="green")
    axes[1].set_title("Trajectory of x1(1)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("x1 value")

    # 3. Average fitness
    axes[2].plot(avg_fitness, color="blue")
    axes[2].set_title("Average Fitness")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Mean Fitness")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
#  TP4 — MULTIPLE PSO RUNS
# ============================================================

def run_multiple_PSO(fitness_func, runs, n_particles, dim, lb, ub,
                     w=0.5, c1=2, c2=2, max_iter=200):
    """
    Run PSO multiple times and compute performance statistics.
    Returns: results dict + all convergence curves.
    """
    all_best_fitness = []
    all_convergence = []
    all_avg_fitness = []
    all_trajectory = []
    all_stagnation = []

    for r in range(runs):
        gBest, gBest_fit, conv, avg_f, traj, stag = PSO(
            fitness_func, n_particles, dim, lb, ub,
            w=w, c1=c1, c2=c2, max_iter=max_iter
        )
        all_best_fitness.append(gBest_fit)
        all_convergence.append(conv)
        all_avg_fitness.append(avg_f)
        all_trajectory.append(traj)
        all_stagnation.append(stag)

    results = np.array(all_best_fitness)

    # Pad convergence curves to same length for averaging
    max_len = max(len(c) for c in all_convergence)
    padded_conv = []
    for c in all_convergence:
        padded = c + [c[-1]] * (max_len - len(c))
        padded_conv.append(padded)

    padded_avg = []
    for a in all_avg_fitness:
        padded = a + [a[-1]] * (max_len - len(a))
        padded_avg.append(padded)

    padded_traj = []
    for t in all_trajectory:
        padded = t + [t[-1]] * (max_len - len(t))
        padded_traj.append(padded)

    mean_convergence = np.mean(padded_conv, axis=0)
    mean_avg_fitness = np.mean(padded_avg, axis=0)
    mean_trajectory = np.mean(padded_traj, axis=0)

    return {
        "Best": np.min(results),
        "Worst": np.max(results),
        "AVG": np.mean(results),
        "STD": np.std(results, ddof=1),
        "Mean Stagnation": np.mean(all_stagnation),
        "mean_convergence": mean_convergence,
        "mean_avg_fitness": mean_avg_fitness,
        "mean_trajectory": mean_trajectory,
    }


# ============================================================
#  TP5 & TP6 — FEATURE SELECTION WITH PSO
# ============================================================

def fitness_feature_selection_topSF(solution, X_train, X_test, y_train, y_test,
                                     SF, alpha=0.9):
    """
    TP5: Fitness function for feature selection.
    Selects the top SF features (highest values in solution vector).
    """
    D = len(solution)

    # Select the SF features with the highest weights
    selected_indices = np.argsort(solution)[-SF:]

    # Train KNN with only the selected features
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train[:, selected_indices], y_train)
    y_pred = knn.predict(X_test[:, selected_indices])

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    ratio = SF / D

    fitness = alpha * error + (1 - alpha) * ratio
    return fitness, accuracy, selected_indices


def fitness_feature_selection_threshold(solution, X_train, X_test, y_train, y_test,
                                         alpha=0.9):
    """
    TP6: Fitness function using threshold > 0.5.
    Features with value > 0.5 are selected automatically.
    """
    D = len(solution)
    selected_indices = np.where(solution > 0.5)[0]

    if len(selected_indices) == 0:
        return 1.0, 0.0, np.array([])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train[:, selected_indices], y_train)
    y_pred = knn.predict(X_test[:, selected_indices])

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    ratio = len(selected_indices) / D

    fitness = alpha * error + (1 - alpha) * ratio
    return fitness, accuracy, selected_indices


def PSO_feature_selection(X_train, X_test, y_train, y_test,
                          SF=None, alpha=0.9,
                          n_particles=30, max_iter=100,
                          w=0.5, c1=2, c2=2, k=0.2,
                          use_threshold=False):
    """
    PSO for feature selection.
    - SF: number of features to select (TP5 mode, top-SF)
    - use_threshold=True: TP6 mode (threshold > 0.5)
    """
    dim = X_train.shape[1]  # number of features
    lb, ub = 0, 1
    vMax = k * (ub - lb)

    # Initialize
    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))

    # Evaluate initial population
    pBest = X.copy()
    pBest_fit = np.full(n_particles, np.inf)
    gBest = None
    gBest_fit = np.inf
    gBest_accuracy = 0.0
    gBest_features = np.array([])

    for i in range(n_particles):
        if use_threshold:
            fit, acc, sel = fitness_feature_selection_threshold(
                X[i], X_train, X_test, y_train, y_test, alpha)
        else:
            fit, acc, sel = fitness_feature_selection_topSF(
                X[i], X_train, X_test, y_train, y_test, SF, alpha)

        pBest_fit[i] = fit
        if fit < gBest_fit:
            gBest = X[i].copy()
            gBest_fit = fit
            gBest_accuracy = acc
            gBest_features = sel.copy()

    convergence = [gBest_fit]
    accuracy_history = [gBest_accuracy]
    stagnation_counter = 0

    # Main loop
    for t in range(max_iter):
        for i in range(n_particles):
            if not np.array_equal(X[i], gBest):
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)

                V[i] = (w * V[i]
                        + c1 * r1 * (pBest[i] - X[i])
                        + c2 * r2 * (gBest - X[i]))
                V[i] = np.clip(V[i], -vMax, vMax)
                X[i] = X[i] + V[i]
                X[i] = np.clip(X[i], lb, ub)

        # Evaluate
        old_gBest_fit = gBest_fit
        for i in range(n_particles):
            if use_threshold:
                fit, acc, sel = fitness_feature_selection_threshold(
                    X[i], X_train, X_test, y_train, y_test, alpha)
            else:
                fit, acc, sel = fitness_feature_selection_topSF(
                    X[i], X_train, X_test, y_train, y_test, SF, alpha)

            # Update personal best
            if fit < pBest_fit[i]:
                pBest[i] = X[i].copy()
                pBest_fit[i] = fit

            # Update global best
            if fit < gBest_fit:
                gBest = X[i].copy()
                gBest_fit = fit
                gBest_accuracy = acc
                gBest_features = sel.copy()

        convergence.append(gBest_fit)
        accuracy_history.append(gBest_accuracy)

        # Stagnation check
        if gBest_fit == old_gBest_fit:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        if stagnation_counter >= 3:
            break

        print(f"  Iter {t+1}: fitness={gBest_fit:.4f}, "
              f"accuracy={gBest_accuracy:.4f}, "
              f"features={len(gBest_features)}")

    return {
        "best_solution": gBest,
        "best_fitness": gBest_fit,
        "best_accuracy": gBest_accuracy,
        "selected_features": sorted(gBest_features),
        "n_selected": len(gBest_features),
        "convergence": convergence,
        "accuracy_history": accuracy_history,
    }


# ============================================================
#  MAIN — Run everything
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TP1 — BENCHMARK FUNCTIONS TEST")
    print("=" * 60)

    # Test with random solutions
    for name, cfg in FUNCTIONS.items():
        x = np.random.uniform(cfg["lb"], cfg["ub"], cfg["dim"])
        val = cfg["func"](x)
        print(f"  {name}: f(random) = {val:.4f}")
    print()

    # ----------------------------------------------------------
    print("=" * 60)
    print("  TP2 — POPULATION EVALUATION")
    print("=" * 60)

    # Example: load or generate a population for F1
    pop = generate_population(30, 30, -100, 100)
    fitness, best_i, worst_i = evaluate_population(pop, F1)
    print(f"  F1 population: Best = {fitness[best_i]:.2f}, "
          f"Worst = {fitness[worst_i]:.2f}")

    # Multiple runs
    stats = run_multiple_populations(F1, runs=50, pop_size=30,
                                     dim=30, lb=-100, ub=100)
    print(f"  F1 (50 runs): Best={stats['Best']:.2f}, "
          f"Worst={stats['Worst']:.2f}, "
          f"AVG={stats['AVG']:.2f}, STD={stats['STD']:.2f}")
    print()

    # ----------------------------------------------------------
    print("=" * 60)
    print("  TP3 — PSO (single run on F1)")
    print("=" * 60)

    gBest, gBest_fit, conv, avg_f, traj, stag = PSO(
        F1, n_particles=30, dim=30, lb=-100, ub=100,
        w=0.5, c1=2, c2=2, max_iter=200
    )
    print(f"  Best fitness: {gBest_fit:.6f}")
    print(f"  Stagnation at iteration: {stag}")
    plot_pso_results(conv, avg_f, traj, title="PSO on F1")
    print()

    # ----------------------------------------------------------
    print("=" * 60)
    print("  TP4 — MULTIPLE PSO RUNS")
    print("=" * 60)

    # Test 1: c1=c2=2
    print("  Running 50 PSO runs with c1=c2=2 ...")
    res1 = run_multiple_PSO(F1, runs=50, n_particles=30, dim=30,
                            lb=-100, ub=100, c1=2, c2=2)
    print(f"  c1=c2=2: Best={res1['Best']:.4f}, Worst={res1['Worst']:.4f}, "
          f"AVG={res1['AVG']:.4f}, STD={res1['STD']:.4f}")

    # Test 2: c1=c2=1
    print("  Running 50 PSO runs with c1=c2=1 ...")
    res2 = run_multiple_PSO(F1, runs=50, n_particles=30, dim=30,
                            lb=-100, ub=100, c1=1, c2=1)
    print(f"  c1=c2=1: Best={res2['Best']:.4f}, Worst={res2['Worst']:.4f}, "
          f"AVG={res2['AVG']:.4f}, STD={res2['STD']:.4f}")

    # Plot mean convergence comparison
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(res1["mean_convergence"], label="c1=c2=2", color="red")
    plt.plot(res2["mean_convergence"], label="c1=c2=1", color="blue")
    plt.title("Mean Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Best Fitness")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(res1["mean_avg_fitness"], label="c1=c2=2", color="red")
    plt.plot(res2["mean_avg_fitness"], label="c1=c2=1", color="blue")
    plt.title("Mean Population Avg Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Avg Fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print()

    # ----------------------------------------------------------
    print("=" * 60)
    print("  TP5 — FEATURE SELECTION (verify fitness function)")
    print("=" * 60)

    # Load digits dataset
    digits = load_digits()
    X_data, y_data = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42
    )

    # Baseline: KNN with ALL 64 features
    knn_all = KNeighborsClassifier(n_neighbors=5)
    knn_all.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, knn_all.predict(X_test))
    print(f"  Baseline (all 64 features): accuracy = {baseline_acc:.4f}")

    # Test Case 1 from teacher: SF=25, alpha=0.9
    solution_case1 = np.array([
        0.74, 0.56, 0.79, 0.92, 0.28, 0.13, 0.53, 0.80, 0.49, 0.91,
        0.91, 0.88, 0.71, 0.96, 0.31, 0.30, 0.01, 0.14, 0.36, 0.42,
        0.53, 0.99, 0.73, 0.53, 0.84, 0.10, 0.34, 0.63, 0.02, 0.29,
        0.46, 0.30, 0.18, 0.21, 0.23, 0.78, 0.59, 0.50, 0.27, 0.30,
        0.36, 0.99, 0.15, 0.60, 0.03, 0.37, 0.52, 0.12, 0.32, 0.69,
        0.48, 0.91, 0.45, 0.57, 0.46, 0.62, 0.68, 0.48, 0.27, 0.94,
        0.47, 0.70, 0.12, 0.35
    ])
    fit1, acc1, sel1 = fitness_feature_selection_topSF(
        solution_case1, X_train, X_test, y_train, y_test, SF=25, alpha=0.9
    )
    print(f"\n  Case 1 (SF=25, alpha=0.9):")
    print(f"    Fitness  = {fit1:.4f}")
    print(f"    Accuracy = {acc1:.4f}")
    print(f"    Selected features: {sorted(sel1)}")

    # Test Case 2 from teacher: SF=10, alpha=0.9
    solution_case2 = np.array([
        0.80, 0.70, 0.89, 0.55, 0.78, 0.63, 0.36, 0.83, 0.18, 0.94,
        0.31, 0.22, 0.53, 0.69, 0.41, 0.52, 0.55, 0.23, 0.74, 0.73,
        0.82, 0.45, 0.35, 0.67, 0.12, 0.62, 0.38, 0.93, 0.04, 0.54,
        0.72, 0.09, 0.23, 0.36, 0.21, 0.56, 0.07, 0.37, 0.60, 0.31,
        0.73, 0.24, 0.71, 0.46, 0.94, 0.17, 0.00, 0.65, 0.48, 0.19,
        0.34, 0.15, 0.42, 0.52, 0.31, 0.29, 0.34, 0.99, 0.59, 0.76,
        0.32, 0.55, 0.16, 0.39
    ])
    fit2, acc2, sel2 = fitness_feature_selection_topSF(
        solution_case2, X_train, X_test, y_train, y_test, SF=10, alpha=0.9
    )
    print(f"\n  Case 2 (SF=10, alpha=0.9):")
    print(f"    Fitness  = {fit2:.4f}")
    print(f"    Accuracy = {acc2:.4f}")
    print(f"    Selected features: {sorted(sel2)}")

    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("  TP5 — RUN PSO FOR FEATURE SELECTION")
    print("=" * 60)

    print("\n  Running PSO feature selection (SF=25, digits dataset)...")
    result_fs = PSO_feature_selection(
        X_train, X_test, y_train, y_test,
        SF=25, alpha=0.9,
        n_particles=20, max_iter=50,
        w=0.5, c1=2, c2=2
    )
    print(f"\n  PSO Results:")
    print(f"    Best fitness:      {result_fs['best_fitness']:.4f}")
    print(f"    Best accuracy:     {result_fs['best_accuracy']:.4f}")
    print(f"    Features selected: {result_fs['n_selected']}")
    print(f"    Feature indices:   {result_fs['selected_features']}")
    print(f"    vs Baseline:       {baseline_acc:.4f} (all 64 features)")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result_fs["convergence"], color="red")
    axes[0].set_title("Fitness Convergence")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Fitness")

    axes[1].plot(result_fs["accuracy_history"], color="green")
    axes[1].set_title("Accuracy Over Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].axhline(y=baseline_acc, color="gray", linestyle="--",
                     label=f"Baseline ({baseline_acc:.3f})")
    axes[1].legend()
    plt.suptitle("PSO Feature Selection (digits, SF=25)", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("  TP6 — FEATURE SELECTION WITH THRESHOLD > 0.5")
    print("=" * 60)

    print("\n  Running PSO feature selection (threshold mode, digits)...")
    result_tp6 = PSO_feature_selection(
        X_train, X_test, y_train, y_test,
        alpha=0.9,
        n_particles=20, max_iter=50,
        w=0.5, c1=2, c2=2,
        use_threshold=True
    )
    print(f"\n  PSO Results (threshold > 0.5):")
    print(f"    Best fitness:      {result_tp6['best_fitness']:.4f}")
    print(f"    Best accuracy:     {result_tp6['best_accuracy']:.4f}")
    print(f"    Features selected: {result_tp6['n_selected']} / 64")
    print(f"    Feature indices:   {result_tp6['selected_features']}")

    print("\n  Done! All TPs completed.")