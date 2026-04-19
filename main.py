"""
TP - Metaheuristics Optimization Algorithms
Main Streamlit Application

This application demonstrates various metaheuristic algorithms (PSO, GA) applied to
optimization benchmarks and feature selection problems.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import modular components
from benchmark_functions import FUNCTIONS, evaluate_population
from pso import PSO, run_multiple_PSO, PSO_feature_selection
from genetic_algorithm import GA_feature_selection
from utils import (
    get_dataset, compute_grid, create_contour_plot,
    create_convergence_plot, pad_convergence_curves, aggregate_results
)


# ============================================================
#  APP CONFIGURATION
# ============================================================

st.set_page_config(page_title="TP - Metaheuristics", layout="wide")
st.markdown("# TP - Metaheuristics")


# ============================================================
#  PART 1: SINGLE SOLUTION OPTIMIZATION
# ============================================================

st.markdown("## Part 1 — Optimization Problem Initialization")
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
    st.markdown(""); st.markdown("")
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
    val = " | ".join([f"{v:.2f}" for v in st.session_state.solution]) if st.session_state.solution is not None else ""
    st.text_area("sol_disp", value=val, height=68, disabled=True, label_visibility="collapsed")
with col_fit:
    st.markdown("Fitness")
    fval = f"{st.session_state.fitness:.2f}" if st.session_state.fitness is not None else ""
    st.text_input("fit_disp", value=fval, disabled=True, label_visibility="collapsed")

st.markdown("")
col_func, col_formula, col_eval = st.columns([1, 3, 1])
with col_func:
    st.markdown("**Function:**")
    selected_func = st.selectbox("func_sel", list(FUNCTIONS.keys()), label_visibility="collapsed")
with col_formula:
    st.latex(FUNCTIONS[selected_func]["latex"])
with col_eval:
    st.markdown(""); st.markdown("")
    evaluate_btn = st.button("Evaluate solution", use_container_width=True)

if evaluate_btn:
    if st.session_state.solution is not None:
        st.session_state.fitness = FUNCTIONS[selected_func]["func"](st.session_state.solution)
        st.rerun()
    else:
        st.warning("Please generate a solution first.")

st.divider()


# ============================================================
#  PART 2: POPULATION INITIALIZATION & EVALUATION
# ============================================================

st.markdown("## Part 2 — Population Initialization & Evaluation")

col_pl, col_ps, col_pb = st.columns([1, 3, 2])
with col_pl:
    st.markdown("**Population:**")
with col_ps:
    pop_size = st.slider("Size", min_value=5, max_value=100, value=30, step=1)
with col_pb:
    gen_pop_btn = st.button("Generate population", use_container_width=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if "population" not in st.session_state:
    st.session_state.population = None
    st.session_state.pop_evaluated = False
    st.session_state.pop_fitness = None
    st.session_state.pop_best_idx = 0

if gen_pop_btn:
    cfg = FUNCTIONS[selected_func]
    st.session_state.population = np.random.uniform(cfg["lb"], cfg["ub"], (pop_size, int(dim)))
    st.session_state.pop_evaluated = False

if uploaded_file is not None:
    try:
        data = np.loadtxt(uploaded_file, delimiter=";")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        st.session_state.population = data
        st.session_state.pop_evaluated = False
        st.success(f"Loaded: {data.shape[0]} solutions × {data.shape[1]} dimensions")
    except Exception as e:
        st.error(f"Error: {e}")

col_eb, col_ei = st.columns([1, 3])
with col_eb:
    eval_pop_btn = st.button("Evaluate population", use_container_width=True)
with col_ei:
    st.markdown(f"**Function ({selected_func})**")

if eval_pop_btn and st.session_state.population is not None:
    cfg = FUNCTIONS[selected_func]
    fitness, bi, wi = evaluate_population(st.session_state.population, cfg["func"])
    st.session_state.pop_fitness = fitness
    st.session_state.pop_best_idx = bi
    st.session_state.pop_evaluated = True

if st.session_state.pop_evaluated and st.session_state.population is not None:
    pop = st.session_state.population
    cfg = FUNCTIONS[selected_func]
    fa = st.session_state.pop_fitness
    bi = st.session_state.pop_best_idx
    st.success(f"**Best** — {fa[bi]:.2f}, **Worst** — {np.max(fa):.2f}, "
               f"**Mean** — {np.mean(fa):.2f}, **STD** — {np.std(fa, ddof=1):.2f}")
    
    Xg, Yg, Zg = compute_grid(cfg["func"], cfg["lb"], cfg["ub"], 100)
    c3d, c2d = st.columns(2)
    with c3d:
        fig = create_contour_plot(Xg, Yg, Zg)  # 3D surface plot
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.7)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        st.pyplot(fig)
        plt.close()
    with c2d:
        fig, ax = create_contour_plot(Xg, Yg, Zg, pop=pop, best_idx=bi, 
                                       title=f"Search Space ({selected_func})")
        st.pyplot(fig)
        plt.close()

st.divider()

# --- Multiple Populations ---
st.markdown("### Running Multiple Populations")
col_rs, col_rb = st.columns([3, 1])
with col_rs:
    n_runs = st.slider("Run", min_value=1, max_value=100, value=15, step=1)
with col_rb:
    eval_runs_btn = st.button("Evaluate", key="tp2_runs", use_container_width=True)

if "runs_results" not in st.session_state:
    st.session_state.runs_results = None

if eval_runs_btn:
    cfg = FUNCTIONS[selected_func]
    bests = []
    all_pops = []
    for _ in range(n_runs):
        rp = np.random.uniform(cfg["lb"], cfg["ub"], (pop_size, int(dim)))
        f, bi, wi = evaluate_population(rp, cfg["func"])
        bests.append(f[bi])
        all_pops.append(rp)
    
    bests = np.array(bests)
    agg_stats = aggregate_results(bests)
    st.session_state.runs_results = {
        **agg_stats,
        "all_pops": all_pops
    }

if st.session_state.runs_results is not None:
    r = st.session_state.runs_results
    st.success(f"**Best** — {r['Best']:.2f}, **Worst** — {r['Worst']:.2f}, "
               f"**Mean** — {r['AVG']:.2f}, **STD** — {r['STD']:.2f}")

st.divider()


# ============================================================
#  PART 3: PSO FOR CONTINUOUS OPTIMIZATION
# ============================================================

st.markdown("## Part 3 — Particle Swarm Optimization")
st.divider()

col_pso_l, col_pso_r = st.columns(2)

with col_pso_l:
    st.markdown("**PSO Parameters**")
    pso_w = st.number_input("Inertia (w)", value=0.5, format="%.2f", step=0.1, key="pso_w")
    pso_c1 = st.number_input("Cognitive (c₁)", value=2.0, format="%.2f", step=0.1, key="pso_c1")
    pso_c2 = st.number_input("Social (c₂)", value=2.0, format="%.2f", step=0.1, key="pso_c2")

with col_pso_r:
    st.markdown("**Metaheuristic Parameters**")
    pso_pop = st.slider("Population (N)", 5, 100, 30, key="pso_pop")
    pso_iter = st.slider("Max Iteration (T)", 10, 500, 200, key="pso_iter")
    pso_runs = st.slider("Run", 1, 50, 15, key="pso_runs")
    pso_eval_btn = st.button("Evaluation", key="pso_eval", use_container_width=True)

if "pso_result" not in st.session_state:
    st.session_state.pso_result = None

if pso_eval_btn:
    with st.spinner(f"Running {pso_runs} PSO experiments..."):
        cfg = FUNCTIONS[selected_func]
        all_best = []
        all_conv = []
        all_avg = []
        all_traj = []
        
        for _ in range(pso_runs):
            res = PSO(cfg["func"], pso_pop, int(dim), cfg["lb"], cfg["ub"],
                     w=pso_w, c1=pso_c1, c2=pso_c2, max_iter=pso_iter)
            all_best.append(res["gBest_fit"])
            all_conv.append(res["convergence"])
            all_avg.append(res["avg_fitness"])
            all_traj.append(res["trajectory"])
        
        all_best = np.array(all_best)
        padded_conv = pad_convergence_curves(all_conv)
        padded_avg = pad_convergence_curves(all_avg)
        padded_traj = pad_convergence_curves(all_traj)
        
        st.session_state.pso_result = {
            "Best": np.min(all_best),
            "Worst": np.max(all_best),
            "AVG": np.mean(all_best),
            "STD": np.std(all_best, ddof=1),
            "mean_conv": np.mean(padded_conv, axis=0),
            "mean_avg": np.mean(padded_avg, axis=0),
            "mean_traj": np.mean(padded_traj, axis=0)
        }

if st.session_state.pso_result is not None:
    r = st.session_state.pso_result
    st.success(f"**Best** — {r['Best']:.4f}, **Worst** — {r['Worst']:.4f}, "
               f"**Mean** — {r['AVG']:.4f}, **STD** — {r['STD']:.4f}")
    
    cc, ct, ca = st.columns(3)
    with cc:
        fig, ax = create_convergence_plot(r["mean_conv"], "Convergence Curve", color="red")
        st.pyplot(fig)
        plt.close()
    with ct:
        fig, ax = create_convergence_plot(r["mean_traj"], "Trajectory of 1st Solution", color="green")
        st.pyplot(fig)
        plt.close()
    with ca:
        fig, ax = create_convergence_plot(r["mean_avg"], "Average Population Fitness", color="blue")
        st.pyplot(fig)
        plt.close()

st.divider()


# ============================================================
#  PART 6: PSO FOR FEATURE SELECTION
# ============================================================

st.markdown("## Part 6 — Feature Selection with PSO")
st.divider()

col_fs_l, col_fs_r = st.columns(2)

with col_fs_l:
    st.markdown("**Feature Selection Parameters**")
    fs_dataset = st.radio("Dataset", ["Synthetic", "Digits"], horizontal=True, key="fs_data")
    fs_alpha = st.number_input("α (accuracy weight)", value=0.99, format="%.2f", step=0.01, key="fs_alpha")
    st.markdown("**PSO Parameters**")
    fs_w = st.number_input("w", value=0.5, format="%.2f", step=0.1, key="fs_w")
    fs_c1 = st.number_input("c₁", value=2.0, format="%.2f", step=0.1, key="fs_c1")
    fs_c2 = st.number_input("c₂", value=2.0, format="%.2f", step=0.1, key="fs_c2")

with col_fs_r:
    st.markdown("**Metaheuristic Parameters**")
    fs_pop = st.slider("Population (N)", 5, 50, 10, key="fs_pop")
    fs_iter = st.slider("Max Iteration (T)", 5, 200, 20, key="fs_iter")
    fs_runs = st.slider("Runs", 1, 50, 15, key="fs_runs")
    fs_eval_btn = st.button("Evaluation", key="fs_eval", use_container_width=True)

X_train, X_test, y_train, y_test = get_dataset(fs_dataset)

if "fs_result" not in st.session_state:
    st.session_state.fs_result = None

if fs_eval_btn:
    with st.spinner(f"Running {fs_runs} PSO feature selection experiments..."):
        all_best = []
        all_acc = []
        all_sel = []
        all_conv = []
        all_avg = []
        
        for _ in range(fs_runs):
            res = PSO_feature_selection(X_train, X_test, y_train, y_test,
                n_particles=fs_pop, max_iter=fs_iter, w=fs_w, c1=fs_c1, c2=fs_c2, alpha=fs_alpha)
            all_best.append(res["gBest_fit"])
            all_acc.append(res["gBest_acc"])
            all_sel.append(res["n_selected"])
            all_conv.append(res["convergence"])
            all_avg.append(res["avg_fitness"])
        
        all_best = np.array(all_best)
        padded_conv = pad_convergence_curves(all_conv)
        padded_avg = pad_convergence_curves(all_avg)
        best_idx = np.argmin(all_best)
        
        st.session_state.fs_result = {
            "Best": np.min(all_best),
            "AVG": np.mean(all_best),
            "STD": np.std(all_best, ddof=1),
            "best_acc": all_acc[best_idx],
            "best_sel": all_sel[best_idx],
            "mean_conv": np.mean(padded_conv, axis=0),
            "mean_avg": np.mean(padded_avg, axis=0)
        }

if st.session_state.fs_result is not None:
    r = st.session_state.fs_result
    st.success(f"**Best Error** — {r['Best']:.4f}, **Mean Error** — {r['AVG']:.4f}, "
               f"**Accuracy** — {r['best_acc']:.2f}, **Features Selected** — {r['best_sel']}, "
               f"**STD** — {r['STD']:.4f}")
    
    cc, ca = st.columns(2)
    with cc:
        fig, ax = create_convergence_plot(r["mean_conv"], "Convergence Curve", color="red")
        st.pyplot(fig)
        plt.close()
    with ca:
        fig, ax = create_convergence_plot(r["mean_avg"], "Average Fitness", color="blue")
        st.pyplot(fig)
        plt.close()

st.divider()


# ============================================================
#  PART 7: GA FOR FEATURE SELECTION
# ============================================================

st.markdown("## Part 7 — Feature Selection with Genetic Algorithm")
st.divider()

col_ga_l, col_ga_r = st.columns(2)

with col_ga_l:
    st.markdown("**Feature Selection Parameters**")
    ga_dataset = st.radio("Dataset", ["Synthetic", "Digits"], horizontal=True, key="ga_data")
    ga_alpha = st.number_input("α (accuracy weight)", value=0.99, format="%.2f", step=0.01, key="ga_alpha")
    st.markdown("**GA Parameters**")
    ga_rc = st.number_input("Crossover Rate (Rc)", value=0.70, format="%.2f", step=0.05, key="ga_rc")
    ga_rm = st.number_input("Mutation Rate (Rm)", value=0.10, format="%.2f", step=0.01, key="ga_rm")

with col_ga_r:
    st.markdown("**Metaheuristic Parameters**")
    ga_pop = st.slider("Population (N)", 5, 50, 10, key="ga_pop")
    ga_iter = st.slider("Max Iteration (T)", 5, 200, 20, key="ga_iter")
    ga_runs = st.slider("Runs", 1, 50, 15, key="ga_runs")
    ga_eval_btn = st.button("Evaluation", key="ga_eval", use_container_width=True)

X_train_ga, X_test_ga, y_train_ga, y_test_ga = get_dataset(ga_dataset)

if "ga_result" not in st.session_state:
    st.session_state.ga_result = None

if ga_eval_btn:
    with st.spinner(f"Running {ga_runs} GA feature selection experiments..."):
        all_best = []
        all_acc = []
        all_sel = []
        all_conv = []
        all_avg = []
        
        for _ in range(ga_runs):
            res = GA_feature_selection(X_train_ga, X_test_ga, y_train_ga, y_test_ga,
                N=ga_pop, T=ga_iter, Rc=ga_rc, Rm=ga_rm, alpha=ga_alpha)
            all_best.append(res["gBest_fit"])
            all_acc.append(res["gBest_acc"])
            all_sel.append(res["n_selected"])
            all_conv.append(res["convergence"])
            all_avg.append(res["avg_fitness"])
        
        all_best = np.array(all_best)
        padded_conv = pad_convergence_curves(all_conv)
        padded_avg = pad_convergence_curves(all_avg)
        best_idx = np.argmin(all_best)
        
        st.session_state.ga_result = {
            "Best": np.min(all_best),
            "AVG": np.mean(all_best),
            "STD": np.std(all_best, ddof=1),
            "best_acc": all_acc[best_idx],
            "best_sel": all_sel[best_idx],
            "mean_conv": np.mean(padded_conv, axis=0),
            "mean_avg": np.mean(padded_avg, axis=0)
        }

if st.session_state.ga_result is not None:
    r = st.session_state.ga_result
    st.success(f"**Best Error** — {r['Best']:.4f}, **Mean Error** — {r['AVG']:.4f}, "
               f"**Accuracy** — {r['best_acc']:.2f}, **Features Selected** — {r['best_sel']}, "
               f"**STD** — {r['STD']:.4f}")
    
    cc, ca = st.columns(2)
    with cc:
        fig, ax = create_convergence_plot(r["mean_conv"], "Convergence Curve", color="red")
        st.pyplot(fig)
        plt.close()
    with ca:
        fig, ax = create_convergence_plot(r["mean_avg"], "Average Fitness", color="blue")
        st.pyplot(fig)
        plt.close()

st.divider()

# Footer
st.markdown("---")
st.markdown("*TP - Metaheuristics | Master 2 SII | USTHB | Prof. I. KHENNAK | 2025/2026*")
