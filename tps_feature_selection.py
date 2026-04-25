"""
TP N°5, N°6, N°7, N°8 pages — Feature Selection with PSO and GA.
MÉTA - Master 2 SII - USTHB
"""
import numpy as np
import pandas as pd
import streamlit as st

from feature_selection import load_dataset, FSEvaluator
from pso import pso
from ga_binary import ga_binary, SELECTION_OPS, CROSSOVER_OPS, REPLACEMENT_OPS
from plotting import line_plot


# =============================================================================
# Shared helpers
# =============================================================================
@st.cache_data(show_spinner=False)
def _cached_dataset(name):
    return load_dataset(name)


def _format_indices(idx):
    return " | ".join(str(i) for i in idx)


def _format_solution(x, n=None):
    """Format a solution — truncate if too long for display."""
    x = np.asarray(x)
    if n is not None and len(x) > n:
        tail = " | …"
        x = x[:n]
    else:
        tail = ""
    return " | ".join(f"{v:.2f}" for v in x) + tail


def _draw_result_curves(res):
    """Three-panel convergence / trajectory / average fitness plot row."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(line_plot(res["convergence_curve"],
                            title="Convergence Curve", color="red"),
                  use_container_width=True)
    with c2:
        st.pyplot(line_plot(res["trajectory_first"],
                            title="Trajectory of 1st solution",
                            ylabel=r"$x_1^{(1)}$", color="limegreen"),
                  use_container_width=True)
    with c3:
        st.pyplot(line_plot(res["average_fitness_curve"],
                            title="Average Fitness of population",
                            color="royalblue"),
                  use_container_width=True)


# =============================================================================
# TP N°5 - Feature Selection with PSO - Part 1
# =============================================================================
def page_tp5():
    st.title("TP - Metaheuristics")
    st.subheader("TP N°5 \\ Feature Selection with PSO - Part 1")

    st.markdown("""
    **Solution Representation.** A solution $x$ is a real-valued vector of
    dimension $D$ (number of features). Each $x_i \\in [0, 1]$ represents the
    importance weight of feature $i$.

    **Objective Function:**
    """)
    st.latex(r"f(x) = \alpha \cdot E_R(x) + (1 - \alpha) \cdot F_R(x)")
    st.caption("Where $E_R = 1 - \\mathrm{Accuracy}$ (KNN error) and "
               "$F_R = N_s / N$ (fraction of selected features). "
               "Selected features = top-$N_s$ values of $x$.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            dataset = st.selectbox("Data", ["Synthetic", "Digits"], key="tp5_ds")
        with c2:
            alpha = st.number_input("α", 0.0, 1.0, 0.90, step=0.01, key="tp5_alpha")
        with c3:
            pass

        X, y, D = _cached_dataset(dataset)
        st.caption(f"Dataset loaded — instances: {X.shape[0]}, features: {D}")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_selected = st.number_input(
                "Number of selected features (Ns)",
                1, int(D), min(25, int(D)), key="tp5_ns",
            )
        with c2:
            gen = st.button("🎲 Generate solution", use_container_width=True,
                            key="tp5_gen")
        with c3:
            evaluate = st.button("✅ Evaluate solution", use_container_width=True,
                                 key="tp5_eval")

        if gen or "tp5_sol" not in st.session_state:
            st.session_state["tp5_sol"] = np.random.random(int(D))
            st.session_state.pop("tp5_result", None)

        sol = st.session_state["tp5_sol"]

        st.markdown("**Solution:**")
        st.code(_format_solution(sol, n=None), language=None)

        if evaluate:
            evaluator = FSEvaluator(X, y, alpha=float(alpha),
                                    n_selected=int(n_selected),
                                    mode="continuous")
            f, acc, ns = evaluator.evaluate(sol)
            idx = evaluator.select_indices(sol)
            st.session_state["tp5_result"] = {
                "fitness": f, "accuracy": acc, "n_selected": ns, "indices": idx,
            }

        if "tp5_result" in st.session_state:
            r = st.session_state["tp5_result"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Fitness", f"{r['fitness']:.4f}")
            c2.metric("Accuracy", f"{r['accuracy']:.4f}")
            c3.metric("Selected", r["n_selected"])
            st.markdown("**Indices of selected features:**")
            st.code(_format_indices(r["indices"]), language=None)


# =============================================================================
# TP N°6 - Feature Selection with PSO - Part 2
# =============================================================================
def page_tp6():
    st.title("TP - Metaheuristics")
    st.subheader("TP N°6 \\ Feature Selection with PSO - Part 2")

    st.markdown("""
    **Application of PSO.** During the implementation of PSO for feature
    selection, if an element $x_i$ of a solution $x$ is greater than 0.5,
    then the corresponding feature is selected and used by KNN.
    """)

    with st.container(border=True):
        st.markdown("### Optimization Problem")

        col_left, col_mid, col_right = st.columns([1.2, 1.2, 0.8])

        with col_left:
            st.markdown("**Feature Selection parameters**")
            c1, c2 = st.columns(2)
            with c1:
                dataset = st.selectbox("Data", ["Synthetic", "Digits"],
                                       key="tp6_ds")
            with c2:
                alpha = st.number_input("α", 0.0, 1.0, 0.99, step=0.01,
                                        key="tp6_alpha")

            st.markdown("**PSO parameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                w = st.number_input("w", 0.0, 2.0, 0.5, step=0.1, key="tp6_w")
            with c2:
                c1_param = st.number_input("c1", 0.0, 4.0, 2.0, step=0.1, key="tp6_c1")
            with c3:
                c2_param = st.number_input("c2", 0.0, 4.0, 2.0, step=0.1, key="tp6_c2")

        with col_mid:
            st.markdown("**Metaheuristic parameters**")
            pop_size = st.slider("Population (N)", 5, 100, 10, key="tp6_pop")
            max_iter = st.slider("Max Iteration (T)", 5, 200, 20, key="tp6_T")
            n_runs = st.slider("Run", 1, 50, 15, key="tp6_runs")

        with col_right:
            st.write("")
            st.write("")
            evaluate = st.button("▶️ Evaluation", use_container_width=True,
                                 key="tp6_eval")

        X, y, D = _cached_dataset(dataset)
        st.caption(f"Dataset: {dataset} — {X.shape[0]} instances, {D} features")

        if evaluate:
            evaluator = FSEvaluator(X, y, alpha=float(alpha),
                                    n_selected=None, mode="continuous")

            all_f = []
            all_acc = []
            all_sel = []
            last_res = None
            last_sol = None

            progress = st.progress(0.0)
            for r in range(int(n_runs)):
                res = pso(evaluator, int(D), 0.0, 1.0,
                          pop_size=int(pop_size), max_iter=int(max_iter),
                          w=float(w), c1=float(c1_param), c2=float(c2_param))
                f, acc, ns = evaluator.evaluate(res["best_solution"])
                all_f.append(f)
                all_acc.append(acc)
                all_sel.append(ns)
                last_res = res
                last_sol = res["best_solution"]
                progress.progress((r + 1) / int(n_runs))
            progress.empty()

            st.session_state["tp6_out"] = {
                "f": np.array(all_f), "acc": np.array(all_acc),
                "sel": np.array(all_sel), "res": last_res, "sol": last_sol,
                "evaluator": evaluator,
            }

    if "tp6_out" in st.session_state:
        out = st.session_state["tp6_out"]
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best", f"{out['f'].min():.4f}")
        c2.metric("Mean (avg error)", f"{out['f'].mean():.4f}")
        c3.metric("Accuracy", f"{out['acc'].max():.4f}")
        c4.metric("Selected", int(np.median(out['sel'])))
        c5.metric("STD", f"{out['f'].std(ddof=1) if len(out['f']) > 1 else 0:.4f}")

        idx = out["evaluator"].select_indices(out["sol"])
        st.markdown("**Indices of selected features (last run's best):**")
        st.code(_format_indices(idx), language=None)

        _draw_result_curves(out["res"])


# =============================================================================
# Shared GA Feature Selection page (used by TP7 and TP8)
# =============================================================================
def _render_ga_fs_page(prefix: str, tp_subtitle: str, intro_md: str):
    """
    Shared interface for the binary GA feature-selection pages.
    `prefix` namespaces session_state and widget keys (e.g. "tp7").
    """
    st.title("TP - Metaheuristics")
    st.subheader(tp_subtitle)

    st.markdown(intro_md)

    with st.container(border=True):
        st.markdown("### Feature Selection with GA")
        st.markdown("**Optimization Problem**")

        col_left, col_mid, col_right = st.columns([1.4, 1.2, 0.8])

        with col_left:
            st.markdown("**Feature Selection parameters**")
            c1, c2 = st.columns(2)
            with c1:
                dataset = st.selectbox("Data", ["Synthetic", "Digits"],
                                       key=f"{prefix}_ds")
            with c2:
                alpha = st.number_input("α", 0.0, 1.0, 0.99, step=0.01,
                                        key=f"{prefix}_alpha")

            st.markdown("**GA parameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                selection = st.selectbox("Selection",
                                         list(SELECTION_OPS.keys()),
                                         index=0, key=f"{prefix}_sel")
            with c2:
                crossover = st.selectbox("Crossover",
                                         list(CROSSOVER_OPS.keys()),
                                         index=0, key=f"{prefix}_cx")
            with c3:
                replacement = st.selectbox("Replacement",
                                           list(REPLACEMENT_OPS.keys()),
                                           index=0, key=f"{prefix}_rep")

            c1, c2 = st.columns(2)
            with c1:
                pc = st.number_input("R_C (crossover)", 0.0, 1.0, 0.70,
                                     step=0.05, key=f"{prefix}_pc")
            with c2:
                pm = st.number_input("R_M (mutation)", 0.0, 1.0, 0.10,
                                     step=0.01, key=f"{prefix}_pm")

        with col_mid:
            st.markdown("**Metaheuristic parameters**")
            pop_size = st.slider("Population (N)", 5, 100, 10,
                                 key=f"{prefix}_pop")
            max_iter = st.slider("Max Iteration (T)", 5, 200, 20,
                                 key=f"{prefix}_T")
            n_runs = st.slider("Run", 1, 50, 15, key=f"{prefix}_runs")

        with col_right:
            st.write("")
            st.write("")
            evaluate = st.button("▶️ Evaluation", use_container_width=True,
                                 key=f"{prefix}_eval")

        X, y, D = _cached_dataset(dataset)
        st.caption(f"Dataset: {dataset} — {X.shape[0]} instances, {D} features")

        if evaluate:
            evaluator = FSEvaluator(X, y, alpha=float(alpha),
                                    n_selected=None, mode="binary")

            all_f, all_acc, all_sel = [], [], []
            last_res = None
            last_sol = None
            progress = st.progress(0.0)
            for r in range(int(n_runs)):
                res = ga_binary(
                    evaluator, int(D),
                    pop_size=int(pop_size), max_iter=int(max_iter),
                    pc=float(pc), pm=float(pm),
                    selection=selection,
                    crossover=crossover,
                    replacement=replacement,
                )
                f, acc, ns = evaluator.evaluate(res["best_solution"])
                all_f.append(f)
                all_acc.append(acc)
                all_sel.append(ns)
                last_res = res
                last_sol = res["best_solution"]
                progress.progress((r + 1) / int(n_runs))
            progress.empty()

            st.session_state[f"{prefix}_out"] = {
                "f": np.array(all_f), "acc": np.array(all_acc),
                "sel": np.array(all_sel), "res": last_res, "sol": last_sol,
                "evaluator": evaluator,
            }

    out_key = f"{prefix}_out"
    if out_key in st.session_state:
        out = st.session_state[out_key]
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Best", f"{out['f'].min():.4f}")
        c2.metric("Mean (avg error)", f"{out['f'].mean():.4f}")
        c3.metric("Accuracy", f"{out['acc'].max():.4f}")
        c4.metric("Selected", int(np.median(out['sel'])))
        c5.metric("STD",
                  f"{out['f'].std(ddof=1) if len(out['f']) > 1 else 0:.4f}")

        idx = out["evaluator"].select_indices(out["sol"])
        st.markdown("**Indices of selected features (last run's best):**")
        st.code(_format_indices(idx), language=None)

        _draw_result_curves(out["res"])


# =============================================================================
# TP N°7 - Genetic Algorithm Part 1 (Feature Selection)
# =============================================================================
def page_tp7():
    _render_ga_fs_page(
        prefix="tp7",
        tp_subtitle="TP N°7 \\ Genetic Algorithm for Feature Selection",
        intro_md=(
            "Binary GA for feature selection — each solution is a vector "
            "$x \\in \\{0, 1\\}^D$ where $x_i = 1$ means feature $i$ is "
            "selected and used by KNN.\n\n"
            "Configure the **Selection**, **Crossover**, and **Replacement** "
            "operators below to explore different GA variants."
        ),
    )
