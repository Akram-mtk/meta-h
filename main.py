import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import *


# ============================================================
#  PSO ALGORITHM
# ============================================================

def PSO(fitness_func, n_particles, dim, lb, ub,
        w=0.5, c1=2, c2=2, max_iter=200, k=0.2):
    vMax = k * (ub - lb)

    X = np.random.uniform(lb, ub, (n_particles, dim))
    V = np.zeros((n_particles, dim))

    fitness = np.array([fitness_func(x) for x in X])
    pBest = X.copy()
    pBest_fit = fitness.copy()
    gBest_idx = np.argmin(fitness)
    gBest = X[gBest_idx].copy()
    gBest_fit = fitness[gBest_idx]

    # Save initial population info
    init_best = gBest_fit
    init_worst = np.max(fitness)
    init_pop = X.copy()

    convergence = [gBest_fit]
    avg_fitness_history = [np.mean(fitness)]
    trajectory = [X[0, 0]]
    stagnation_counter = 0
    stagnation_iter = 0

    # Store all positions for scatter plot
    all_positions = [X.copy()]

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

        fitness = np.array([fitness_func(x) for x in X])
        improved = fitness < pBest_fit
        pBest[improved] = X[improved].copy()
        pBest_fit[improved] = fitness[improved]

        best_idx = np.argmin(fitness)
        old_gBest_fit = gBest_fit
        if fitness[best_idx] < gBest_fit:
            gBest = X[best_idx].copy()
            gBest_fit = fitness[best_idx]
            stagnation_iter = t + 1

        convergence.append(gBest_fit)
        avg_fitness_history.append(np.mean(fitness))
        trajectory.append(X[0, 0])
        all_positions.append(X.copy())

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
        "all_positions": all_positions,
    }


# ============================================================
#  HELPER: Compute contour grid
# ============================================================

def compute_grid(func, lb, ub, pts=100):
    x_r = np.linspace(lb, ub, pts)
    y_r = np.linspace(lb, ub, pts)
    Xg, Yg = np.meshgrid(x_r, y_r)
    Zg = np.zeros_like(Xg)
    for i in range(pts):
        for j in range(pts):
            Zg[i, j] = func(np.array([Xg[i, j], Yg[i, j]]))
    return Xg, Yg, Zg







# ============================================================
#  STREAMLIT APP
# ============================================================

st.set_page_config(page_title="TP - Metaheuristics", layout="wide")

st.markdown("# TP - Metaheuristics")

# ============================================================
#  TP1 — OPTIMIZATION PROBLEM INITIALIZATION
# ============================================================

st.markdown("## Part 1 \\ Optimization Problem Initialization")
st.markdown("### Standard Continuous Optimization Benchmark Problems in Metaheuristics")
st.divider()

st.markdown("**Solution:**")
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

with col1:
    dim = st.number_input("Dimension (D)", min_value=1, max_value=1000, value=30, step=1)
with col2:
    lb = st.number_input("Range (min)", value=-100.0, format="%.2f")
with col3:
    ub = st.number_input("Range (max)", value=100.0, format="%.2f")
with col4:
    st.markdown("")
    st.markdown("")
    generate_btn = st.button("Generate solution", use_container_width=True)

if "solution" not in st.session_state:
    st.session_state.solution = None
    st.session_state.fitness = None

if generate_btn:
    st.session_state.solution = np.random.uniform(lb, ub, int(dim))
    st.session_state.fitness = None

col_sol, col_fit = st.columns([4, 1])
with col_sol:
    st.markdown("Candidate solution example")
    if st.session_state.solution is not None:
        sol_str = " | ".join([f"{v:.2f}" for v in st.session_state.solution])
        st.text_area("solution_display", value=sol_str, height=68, disabled=True,
                     label_visibility="collapsed")
    else:
        st.text_area("solution_empty", value="", height=68, disabled=True,
                     label_visibility="collapsed")
with col_fit:
    st.markdown("Fitness")
    if st.session_state.fitness is not None:
        st.text_input("fitness_display", value=f"{st.session_state.fitness:.2f}",
                      disabled=True, label_visibility="collapsed")
    else:
        st.text_input("fitness_empty", value="", disabled=True,
                      label_visibility="collapsed")

st.markdown("")
col_func, col_formula, col_eval = st.columns([1, 3, 1])
with col_func:
    st.markdown("**Function:**")
    selected_func = st.selectbox("func_select", list(FUNCTIONS.keys()),
                                 label_visibility="collapsed")
with col_formula:
    cfg = FUNCTIONS[selected_func]
    st.latex(cfg["latex"])
with col_eval:
    st.markdown("")
    st.markdown("")
    evaluate_btn = st.button("Evaluate solution", use_container_width=True)

if evaluate_btn:
    if st.session_state.solution is not None:
        cfg = FUNCTIONS[selected_func]
        st.session_state.fitness = cfg["func"](st.session_state.solution)
        st.rerun()
    else:
        st.warning("Please generate a solution first.")

st.divider()


# ============================================================
#  TP2 — POPULATION INITIALIZATION
# ============================================================

st.markdown("## Population Initialization")

col_pop_label, col_pop_slider, col_pop_btn = st.columns([1, 3, 2])
with col_pop_label:
    st.markdown("**Population:**")
with col_pop_slider:
    pop_size = st.slider("Size", min_value=5, max_value=100, value=30, step=1)
with col_pop_btn:
    generate_pop_btn = st.button("Generate population", use_container_width=True)

uploaded_file = st.file_uploader("Upload population CSV", type=["csv"],
                                  label_visibility="collapsed")

if "population" not in st.session_state:
    st.session_state.population = None
    st.session_state.pop_evaluated = False
    st.session_state.pop_best_fit = None
    st.session_state.pop_worst_fit = None
    st.session_state.pop_best_idx = 0

if generate_pop_btn:
    cfg = FUNCTIONS[selected_func]
    st.session_state.population = np.random.uniform(cfg["lb"], cfg["ub"],
                                                     (pop_size, int(dim)))
    st.session_state.pop_evaluated = False

if uploaded_file is not None:
    try:
        data = np.loadtxt(uploaded_file, delimiter=";")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        st.session_state.population = data
        st.session_state.pop_evaluated = False
        st.success(f"Loaded population: {data.shape[0]} solutions, {data.shape[1]} dimensions")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

col_ev_btn, col_ev_info = st.columns([1, 3])
with col_ev_btn:
    evaluate_pop_btn = st.button("Evaluate population", use_container_width=True)
with col_ev_info:
    st.markdown(f"**Function ({selected_func})**")

if evaluate_pop_btn:
    if st.session_state.population is not None:
        cfg = FUNCTIONS[selected_func]
        fitness, best_idx, worst_idx = evaluate_population(
            st.session_state.population, cfg["func"]
        )
        st.session_state.pop_fitness = fitness
        st.session_state.pop_best_fit = fitness[best_idx]
        st.session_state.pop_worst_fit = fitness[worst_idx]
        st.session_state.pop_best_idx = best_idx
        st.session_state.pop_evaluated = True
    else:
        st.warning("Please generate or load a population first.")

if st.session_state.pop_evaluated and st.session_state.population is not None:
    pop = st.session_state.population
    cfg = FUNCTIONS[selected_func]
    fit_arr = st.session_state.pop_fitness

    st.success(
        f"**Best** — {st.session_state.pop_best_fit:.2f}, "
        f"**Worst** — {st.session_state.pop_worst_fit:.2f}, "
        f"**Mean** — {np.mean(fit_arr):.2f}, "
        f"**STD** — {np.std(fit_arr, ddof=1):.2f}"
    )

    Xg, Yg, Zg = compute_grid(cfg["func"], cfg["lb"], cfg["ub"], 100)

    col_3d, col_2d = st.columns(2)
    with col_3d:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.7)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        st.pyplot(fig)
        plt.close()
    with col_2d:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.contour(Xg, Yg, Zg, levels=30, cmap="viridis")
        ax2.scatter(pop[:, 0], pop[:, 1], c="black", s=15, label="Solution")
        bi = st.session_state.pop_best_idx
        ax2.scatter(pop[bi, 0], pop[bi, 1], c="red", s=100,
                   marker="o", label="Best solution", zorder=5)
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.set_title(f"Search History ({selected_func})")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()

st.divider()

# ============================================================
#  TP2 — RUNNING MULTIPLE POPULATIONS
# ============================================================

st.markdown("### Running Multiple Populations")

col_run_slider, col_run_btn = st.columns([3, 1])
with col_run_slider:
    n_runs = st.slider("Run", min_value=1, max_value=100, value=15, step=1)
with col_run_btn:
    evaluate_runs_btn = st.button("Evaluate", use_container_width=True)

if "runs_results" not in st.session_state:
    st.session_state.runs_results = None

if evaluate_runs_btn:
    cfg = FUNCTIONS[selected_func]
    best_per_run = []
    all_populations = []
    for r in range(n_runs):
        run_pop = np.random.uniform(cfg["lb"], cfg["ub"], (pop_size, int(dim)))
        fitness, best_idx, worst_idx = evaluate_population(run_pop, cfg["func"])
        best_per_run.append(fitness[best_idx])
        all_populations.append(run_pop)
    best_per_run = np.array(best_per_run)
    st.session_state.runs_results = {
        "Best": np.min(best_per_run),
        "Worst": np.max(best_per_run),
        "AVG": np.mean(best_per_run),
        "STD": np.std(best_per_run, ddof=1),
        "all_populations": all_populations,
    }

if st.session_state.runs_results is not None:
    res = st.session_state.runs_results
    cfg = FUNCTIONS[selected_func]

    col_stats, col_scatter = st.columns([1, 2])
    with col_stats:
        st.markdown(
            f"**Best** — {res['Best']:.2f},\n\n"
            f"**Worst** — {res['Worst']:.2f},\n\n"
            f"**Mean (average error)** — {res['AVG']:.2f},\n\n"
            f"**STD** — {res['STD']:.2f},"
        )
    with col_scatter:
        Xg, Yg, Zg = compute_grid(cfg["func"], cfg["lb"], cfg["ub"], 200)
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.contour(Xg, Yg, Zg, levels=30, cmap="viridis")
        for rp in res["all_populations"]:
            ax3.scatter(rp[:, 0], rp[:, 1], c="black", s=5, alpha=0.3)
        # Find overall best
        gb_val = np.inf
        gb_pos = None
        for rp in res["all_populations"]:
            fit = np.array([cfg["func"](s) for s in rp])
            bi = np.argmin(fit)
            if fit[bi] < gb_val:
                gb_val = fit[bi]
                gb_pos = rp[bi]
        if gb_pos is not None:
            ax3.scatter(gb_pos[0], gb_pos[1], c="red", s=120,
                       marker="o", label="Best solution", zorder=5)
        ax3.set_xlabel("x₁")
        ax3.set_ylabel("x₂")
        ax3.set_title(f"Search History ({selected_func})")
        ax3.legend()
        st.pyplot(fig3)
        plt.close()

st.divider()


# ============================================================
#  TP3 — PARTICLE SWARM OPTIMIZATION
# ============================================================

st.markdown("## Part 3 \\ Particle Swarm Optimization")
st.divider()

# --- Parameters row ---
col_sr_label, col_sr_btn, col_empty, col_meta, col_iter = st.columns([1, 1, 1, 1, 1])

with col_sr_label:
    st.markdown("**Single run:**")
with col_sr_btn:
    pso_run_btn = st.button("Evaluate", key="pso_eval", use_container_width=True)
with col_meta:
    st.markdown("Metaheuristic")
    st.text_input("meta_label", value="PSO", disabled=True, label_visibility="collapsed")
with col_iter:
    max_iter = st.number_input("Max Iteration (T)", min_value=10, max_value=1000,
                                value=200, step=10)

col_w, col_c1, col_c2 = st.columns(3)
with col_w:
    w_val = st.number_input("w", value=0.5, format="%.2f", step=0.1)
with col_c1:
    c1_val = st.number_input("c₁", value=2.0, format="%.2f", step=0.1)
with col_c2:
    c2_val = st.number_input("c₂", value=2.0, format="%.2f", step=0.1)

# Session state for PSO
if "pso_result" not in st.session_state:
    st.session_state.pso_result = None

if pso_run_btn:
    cfg = FUNCTIONS[selected_func]
    with st.spinner("Running PSO..."):
        result = PSO(cfg["func"], pop_size, int(dim), cfg["lb"], cfg["ub"],
                     w=w_val, c1=c1_val, c2=c2_val, max_iter=int(max_iter))
    st.session_state.pso_result = result

# --- Display PSO results ---
if st.session_state.pso_result is not None:
    r = st.session_state.pso_result
    cfg = FUNCTIONS[selected_func]

    st.markdown(f"**Function ({selected_func})**")

    # --- Row 1: 3D surface + 1st iteration scatter + final iteration scatter + results ---
    Xg, Yg, Zg = compute_grid(cfg["func"], cfg["lb"], cfg["ub"], 100)

    col_3d, col_init, col_final, col_res = st.columns([1, 1, 1, 1])

    with col_3d:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.7)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        st.pyplot(fig)
        plt.close()

    with col_init:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.contour(Xg, Yg, Zg, levels=30, cmap="viridis")
        init_p = r["init_pop"]
        ax2.scatter(init_p[:, 0], init_p[:, 1], c="black", s=15, label="Solution")
        # Best of initial pop
        init_fit = np.array([cfg["func"](s) for s in init_p])
        ib = np.argmin(init_fit)
        ax2.scatter(init_p[ib, 0], init_p[ib, 1], c="red", s=80,
                   marker="o", label="Best solution", zorder=5)
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.set_title(f"Search History ({selected_func}), 1st Iteration")
        ax2.legend(fontsize=7)
        st.pyplot(fig2)
        plt.close()

    with col_final:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.contour(Xg, Yg, Zg, levels=30, cmap="viridis")
        # Plot all positions from all iterations
        for pos in r["all_positions"]:
            ax3.scatter(pos[:, 0], pos[:, 1], c="black", s=3, alpha=0.2)
        final_p = r["final_pop"]
        ax3.scatter(final_p[:, 0], final_p[:, 1], c="orange", s=25,
                   label="Best solution at iteration t", zorder=4)
        ax3.scatter(r["gBest"][0], r["gBest"][1], c="red", s=100,
                   marker="o", label="Best solution", zorder=5)
        ax3.set_xlabel("x₁")
        ax3.set_ylabel("x₂")
        ax3.set_title(f"Search History ({selected_func}), Final Iteration")
        ax3.legend(fontsize=7)
        st.pyplot(fig3)
        plt.close()

    with col_res:
        st.markdown("**Initial population:**")
        st.markdown(f"**Best** — {r['init_best']:.2f}, **Worst** — {r['init_worst']:.2f}")
        st.markdown("")
        st.markdown("**Final population:**")
        st.markdown(f"**Best** — {r['gBest_fit']:.4f}")
        st.markdown("")
        st.markdown(f"**Stagnation** — Iteration N°{r['stagnation_iter']}")

    # --- Row 2: Convergence + Trajectory + Average Fitness ---
    col_conv, col_traj, col_avg = st.columns(3)

    with col_conv:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.plot(r["convergence"], color="red")
        ax4.set_title("Convergence Curve")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Fitness")
        st.pyplot(fig4)
        plt.close()

    with col_traj:
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.plot(r["trajectory"], color="green")
        ax5.set_title("Trajectory of 1st solution")
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("x₁⁽¹⁾")
        st.pyplot(fig5)
        plt.close()

    with col_avg:
        fig6, ax6 = plt.subplots(figsize=(5, 3.5))
        ax6.plot(r["avg_fitness"], color="blue")
        ax6.set_title("Average Fitness")
        ax6.set_xlabel("Iteration")
        ax6.set_ylabel("Fitness")
        st.pyplot(fig6)
        plt.close()

st.divider()