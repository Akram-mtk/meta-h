import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from functions import *


# ============================================================
#  STREAMLIT APP
# ============================================================

st.set_page_config(page_title="TP - Metaheuristics", layout="wide")

st.markdown("# TP - Metaheuristics")
st.markdown("## Part 1 \\ Optimization Problem Initialization")
st.markdown("### Standard Continuous Optimization Benchmark Problems in Metaheuristics")

st.divider()

# --- Solution row ---
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

# Session state init
if "solution" not in st.session_state:
    st.session_state.solution = None
    st.session_state.fitness = None

if generate_btn:
    st.session_state.solution = np.random.uniform(lb, ub, int(dim))
    st.session_state.fitness = None

# Display candidate solution and fitness
col_sol, col_fit = st.columns([4, 1])

with col_sol:
    st.markdown("Candidate solution example")
    if st.session_state.solution is not None:
        sol_str = " | ".join([f"{v:.2f}" for v in st.session_state.solution])
        st.text_area("solution_display", value=sol_str, height=68, disabled=True,
                     label_visibility="collapsed")
    else:
        st.text_area("solution_display_empty", value="", height=68, disabled=True,
                     label_visibility="collapsed")

with col_fit:
    st.markdown("Fitness")
    if st.session_state.fitness is not None:
        st.text_input("fitness_display", value=f"{st.session_state.fitness:.2f}",
                      disabled=True, label_visibility="collapsed")
    else:
        st.text_input("fitness_display_empty", value="", disabled=True,
                      label_visibility="collapsed")

st.markdown("")

# --- Function row ---
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
        fitness = cfg["func"](st.session_state.solution)
        st.session_state.fitness = fitness
        st.rerun()
    else:
        st.warning("Please generate a solution first.")

st.divider()





# ============================================================
#  TP2 — POPULATION INITIALIZATION
# ============================================================

st.markdown("## Population Initialization")

# --- Population row: Size slider + Generate button ---
col_pop_label, col_pop_slider, col_pop_btn = st.columns([1, 3, 2])

with col_pop_label:
    st.markdown("**Population:**")

with col_pop_slider:
    pop_size = st.slider("Size", min_value=5, max_value=100, value=30, step=1)

with col_pop_btn:
    generate_pop_btn = st.button("Generate population", use_container_width=True)

# --- CSV file upload ---
uploaded_file = st.file_uploader("Upload population CSV", type=["csv"],
                                  label_visibility="collapsed")

# Session state for population
if "population" not in st.session_state:
    st.session_state.population = None
    st.session_state.pop_evaluated = False
    st.session_state.pop_best_fit = None
    st.session_state.pop_worst_fit = None
    st.session_state.pop_fitness = None
    st.session_state.pop_best_idx = 0

# Generate random population
if generate_pop_btn:
    cfg = FUNCTIONS[selected_func]
    st.session_state.population = np.random.uniform(cfg["lb"], cfg["ub"],
                                                     (pop_size, int(dim)))
    st.session_state.pop_evaluated = False

# Load from CSV
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

# --- Evaluate population button ---
col_ev_btn, col_ev_info = st.columns([1, 3])

with col_ev_btn:
    evaluate_pop_btn = st.button("Evaluate population", use_container_width=True)

with col_ev_info:
    st.markdown(f"**Function ({selected_func})**")

# Evaluate population
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

# --- Display evaluation results ---
if st.session_state.pop_evaluated and st.session_state.population is not None:
    pop = st.session_state.population
    cfg = FUNCTIONS[selected_func]

    st.success(f"**Best** — {st.session_state.pop_best_fit:.2f}, "
               f"**Worst** — {st.session_state.pop_worst_fit:.2f}")

    # --- Plots: 3D surface + 2D contour scatter ---
    col_3d, col_2d = st.columns(2)

    # Compute grid (shared)
    grid_pts = 100
    x_range = np.linspace(cfg["lb"], cfg["ub"], grid_pts)
    y_range = np.linspace(cfg["lb"], cfg["ub"], grid_pts)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = np.zeros_like(X_grid)
    for i in range(grid_pts):
        for j in range(grid_pts):
            Z_grid[i, j] = cfg["func"](np.array([X_grid[i, j], Y_grid[i, j]]))

    with col_3d:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis", alpha=0.7)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        st.pyplot(fig)
        plt.close()

    with col_2d:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.contour(X_grid, Y_grid, Z_grid, levels=30, cmap="viridis")
        ax2.scatter(pop[:, 0], pop[:, 1], c="black", s=15, label="Solution")
        best_i = st.session_state.pop_best_idx
        ax2.scatter(pop[best_i, 0], pop[best_i, 1], c="red", s=100,
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

# Session state for runs
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

# --- Display multiple runs results ---
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
        fig3, ax3 = plt.subplots(figsize=(6, 5))

        grid_pts2 = 200
        x_r = np.linspace(cfg["lb"], cfg["ub"], grid_pts2)
        y_r = np.linspace(cfg["lb"], cfg["ub"], grid_pts2)
        Xg, Yg = np.meshgrid(x_r, y_r)
        Zg = np.zeros_like(Xg)
        for i in range(grid_pts2):
            for j in range(grid_pts2):
                Zg[i, j] = cfg["func"](np.array([Xg[i, j], Yg[i, j]]))

        ax3.contour(Xg, Yg, Zg, levels=30, cmap="viridis")

        # Plot all solutions from all runs
        for run_pop in res["all_populations"]:
            ax3.scatter(run_pop[:, 0], run_pop[:, 1], c="black", s=5, alpha=0.3)

        # Find and plot overall best
        global_best_val = np.inf
        global_best_pos = None
        for run_pop in res["all_populations"]:
            fitness = np.array([cfg["func"](s) for s in run_pop])
            bi = np.argmin(fitness)
            if fitness[bi] < global_best_val:
                global_best_val = fitness[bi]
                global_best_pos = run_pop[bi]

        if global_best_pos is not None:
            ax3.scatter(global_best_pos[0], global_best_pos[1], c="red", s=120,
                       marker="o", label="Best solution", zorder=5)

        ax3.set_xlabel("x₁")
        ax3.set_ylabel("x₂")
        ax3.set_title(f"Search History ({selected_func})")
        ax3.legend()
        st.pyplot(fig3)
        plt.close()

st.divider()