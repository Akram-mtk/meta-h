import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================================
#  FITNESS FUNCTION FOR FEATURE SELECTION (binary encoding)
# ============================================================

def fitness_ga(solution, X_train, X_test, y_train, y_test, alpha=0.99):
    """
    Fitness for GA feature selection.
    solution: binary array (0 or 1) of length D (number of features)
    bit = 1 → feature selected, bit = 0 → feature excluded
    """
    selected = np.where(solution == 1)[0]

    # If no feature selected → worst possible fitness
    if len(selected) == 0:
        return 1.0, 0.0, 0

    D = len(solution)
    SF = len(selected)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train[:, selected], y_train)
    y_pred = knn.predict(X_test[:, selected])

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    ratio = SF / D

    fitness = alpha * error + (1 - alpha) * ratio
    return fitness, accuracy, SF


# ============================================================
#  GENETIC ALGORITHM (following teacher's pseudocode exactly)
# ============================================================

def GA(X_train, X_test, y_train, y_test,
       N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """
    Genetic Algorithm for feature selection.

    Parameters:
        N   — population size
        T   — max iterations (generations)
        Rc  — crossover probability
        Rm  — mutation probability
        alpha — weight for accuracy in fitness function

    Returns: dict with results
    """
    D = X_train.shape[1]  # number of features = number of bits

    # ---- Initialize random binary population ----
    pop = np.random.randint(0, 2, (N, D))

    # Evaluate initial population and find best
    gBest = None
    gBest_fit = np.inf
    gBest_acc = 0.0
    gBest_sf = 0

    fitness_values = np.zeros(N)
    for i in range(N):
        fit, acc, sf = fitness_ga(pop[i], X_train, X_test, y_train, y_test, alpha)
        fitness_values[i] = fit
        if fit < gBest_fit:
            gBest = pop[i].copy()
            gBest_fit = fit
            gBest_acc = acc
            gBest_sf = sf

    convergence = [gBest_fit]
    avg_fitness_history = [np.mean(fitness_values)]
    trajectory = [pop[0][0]]  # track first bit of first solution
    stagnation_counter = 0

    # ---- Main loop ----
    for t in range(T):
        new_pop = []

        # Generate N children (N/2 pairs of parents → N/2 pairs of children)
        for _ in range(N // 2):

            # === SELECTION: randomly pick 2 parents ===
            p1_idx = np.random.randint(N)
            p2_idx = np.random.randint(N)
            parent1 = pop[p1_idx].copy()
            parent2 = pop[p2_idx].copy()

            # === CROSSOVER ===
            r = np.random.random()
            if r < Rc:
                # Choose random crossover point k (between 1 and D-1)
                k = np.random.randint(1, D)
                child1 = np.concatenate([parent1[:k], parent2[k:]])
                child2 = np.concatenate([parent2[:k], parent1[k:]])
            else:
                # No crossover → children are copies of parents
                child1 = parent1.copy()
                child2 = parent2.copy()

            # === MUTATION ===
            for j in range(D):
                r = np.random.random()
                if r < Rm:
                    child1[j] = 1 - child1[j]  # flip: 0→1 or 1→0

                r = np.random.random()
                if r < Rm:
                    child2[j] = 1 - child2[j]

            new_pop.append(child1)
            new_pop.append(child2)

        # === REPLACEMENT: new population replaces old one ===
        pop = np.array(new_pop[:N])

        # Evaluate new population and update best
        old_gBest_fit = gBest_fit
        for i in range(N):
            fit, acc, sf = fitness_ga(pop[i], X_train, X_test, y_train, y_test, alpha)
            fitness_values[i] = fit
            if fit < gBest_fit:
                gBest = pop[i].copy()
                gBest_fit = fit
                gBest_acc = acc
                gBest_sf = sf

        # Track history
        convergence.append(gBest_fit)
        avg_fitness_history.append(np.mean(fitness_values))
        trajectory.append(pop[0][0])

        # Stagnation check
        if gBest_fit == old_gBest_fit:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        if stagnation_counter >= 3:
            break

    return {
        "best_solution": gBest,
        "best_fitness": gBest_fit,
        "best_accuracy": gBest_acc,
        "selected_features": sorted(np.where(gBest == 1)[0]),
        "n_selected": gBest_sf,
        "convergence": convergence,
        "avg_fitness": avg_fitness_history,
        "trajectory": trajectory,
    }


# ============================================================
#  MULTIPLE RUNS (to compute Best, Worst, AVG, STD)
# ============================================================

def run_multiple_GA(X_train, X_test, y_train, y_test,
                    runs=15, N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99):
    """Run GA multiple times and compute performance statistics."""

    all_best_fitness = []
    all_convergence = []
    all_avg_fitness = []
    all_trajectory = []

    for r in range(runs):
        result = GA(X_train, X_test, y_train, y_test,
                    N=N, T=T, Rc=Rc, Rm=Rm, alpha=alpha)
        all_best_fitness.append(result["best_fitness"])
        all_convergence.append(result["convergence"])
        all_avg_fitness.append(result["avg_fitness"])
        all_trajectory.append(result["trajectory"])

        print(f"  Run {r+1}/{runs}: fitness={result['best_fitness']:.4f}, "
              f"accuracy={result['best_accuracy']:.4f}, "
              f"selected={result['n_selected']}")

    results = np.array(all_best_fitness)

    # Pad curves to same length for averaging
    max_len = max(len(c) for c in all_convergence)
    padded_conv = [c + [c[-1]] * (max_len - len(c)) for c in all_convergence]
    padded_avg = [a + [a[-1]] * (max_len - len(a)) for a in all_avg_fitness]
    padded_traj = [t + [t[-1]] * (max_len - len(t)) for t in all_trajectory]

    return {
        "Best": np.min(results),
        "Worst": np.max(results),
        "AVG": np.mean(results),
        "STD": np.std(results, ddof=1),
        "mean_convergence": np.mean(padded_conv, axis=0),
        "mean_avg_fitness": np.mean(padded_avg, axis=0),
        "mean_trajectory": np.mean(padded_traj, axis=0),
    }


# ============================================================
#  PLOTTING
# ============================================================

def plot_ga_results(convergence, avg_fitness, trajectory, title="GA"):
    """Plot the 3 charts required by the teacher."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(convergence, color="red")
    axes[0].set_title("Convergence Curve")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best Fitness")

    axes[1].plot(trajectory, color="green")
    axes[1].set_title("Trajectory of 1st solution")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")

    axes[2].plot(avg_fitness, color="blue")
    axes[2].set_title("Average Fitness of population")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Mean Fitness")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    # ---- Generate Synthetic dataset (as in teacher's slides) ----
    print("=" * 60)
    print("  TP7 — GENETIC ALGORITHM FOR FEATURE SELECTION")
    print("=" * 60)

    X_syn, y_syn = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=5,
        n_redundant=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_syn, y_syn, test_size=0.3, random_state=42
    )

    # ---- Baseline: KNN with ALL features ----
    knn_all = KNeighborsClassifier(n_neighbors=5)
    knn_all.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, knn_all.predict(X_test))
    print(f"\n  Baseline (all 50 features): accuracy = {baseline_acc:.4f}")

    # ---- Single GA run ----
    print("\n  --- Single GA run ---")
    result = GA(X_train, X_test, y_train, y_test,
                N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99)

    print(f"  Best fitness:      {result['best_fitness']:.4f}")
    print(f"  Best accuracy:     {result['best_accuracy']:.4f}")
    print(f"  Features selected: {result['n_selected']} / 50")
    print(f"  Feature indices:   {result['selected_features']}")

    plot_ga_results(result["convergence"], result["avg_fitness"],
                    result["trajectory"], title="GA on Synthetic dataset")

    # ---- Multiple GA runs (as in teacher's slides) ----
    print("\n  --- Multiple GA runs (15 runs) ---")
    stats = run_multiple_GA(
        X_train, X_test, y_train, y_test,
        runs=15, N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99
    )

    print(f"\n  Results over 15 runs:")
    print(f"    Best:  {stats['Best']:.4f}")
    print(f"    Worst: {stats['Worst']:.4f}")
    print(f"    AVG:   {stats['AVG']:.4f}")
    print(f"    STD:   {stats['STD']:.4f}")

    # Plot mean curves
    plot_ga_results(stats["mean_convergence"], stats["mean_avg_fitness"],
                    stats["mean_trajectory"],
                    title="GA Mean Results (15 runs)")

    # ---- Also test on Digits dataset ----
    print("\n" + "=" * 60)
    print("  GA ON DIGITS DATASET")
    print("=" * 60)

    digits = load_digits()
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42
    )

    knn_all_d = KNeighborsClassifier(n_neighbors=5)
    knn_all_d.fit(X_train_d, y_train_d)
    baseline_d = accuracy_score(y_test_d, knn_all_d.predict(X_test_d))
    print(f"\n  Baseline (all 64 features): accuracy = {baseline_d:.4f}")

    result_d = GA(X_train_d, X_test_d, y_train_d, y_test_d,
                  N=10, T=20, Rc=0.70, Rm=0.10, alpha=0.99)

    print(f"  GA fitness:        {result_d['best_fitness']:.4f}")
    print(f"  GA accuracy:       {result_d['best_accuracy']:.4f}")
    print(f"  Features selected: {result_d['n_selected']} / 64")
    print(f"  Feature indices:   {result_d['selected_features']}")

    print("\n  Done!")