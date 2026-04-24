import io
import numpy as np
import pandas as pd
import streamlit as st

from benchmarks import FUNCTIONS, get_function, get_min
from plotting import surface_plot, contour_scatter, line_plot
from pso import pso
from ga import ga


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="TP - Metaheuristics",
    page_icon="🧬",
    layout="wide",
)



# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.title("📘 TP - Metaheuristics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select TP:",
    [
        "TP N°1 – Optimization Problem Initialization",
        "TP N°2 – Population Initialization (Part 2)",
        "TP N°3 – PSO (Part 1)",
        "TP N°4 – PSO (Part 2 - Multiple Runs)",
        "TP N°7 – Genetic Algorithm",
    ],
)


# =============================================================================
# UTILITIES
# =============================================================================
def load_population_from_csv(file) -> np.ndarray:
    """
    Load a population of solutions from a CSV file.
    Supports both ';'-separated (teacher's format: one row, values separated
    by ';' per line, multiple lines for multiple solutions) and ','-separated.
    """
    raw = file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    rows = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Try ';' first, then ','
        parts = line.split(";") if ";" in line else line.split(",")
        try:
            vals = [float(p.strip()) for p in parts if p.strip() != ""]
            if vals:
                rows.append(vals)
        except ValueError:
            # Header line — skip
            continue
    # Pad/truncate rows to the same length
    max_len = max(len(r) for r in rows)
    rows = [r + [0.0] * (max_len - len(r)) for r in rows]
    return np.array(rows, dtype=float)


def format_solution(x: np.ndarray) -> str:
    return " | ".join(f"{v:.2f}" for v in x)


# =============================================================================
# TP N°1 - Optimization Problem Initialization
# =============================================================================
def page_tp1():

    st.title("TP - Metaheuristics")
    st.subheader("Part 1 \\ Optimization Problem Initialization")
    st.markdown("### Standard Continuous Optimization Benchmark Problems in Metaheuristics")

    # --- Function selector first so its range can drive defaults ---
    fn_name = st.selectbox(
        "**Function:**",
        list(FUNCTIONS.keys()),
        index=0,
        key="tp1_fn",
    )
    fn, lb_default, ub_default, info = get_function(fn_name)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
        with c1:
            dim = st.number_input("Dimension (D)", min_value=1, max_value=1000,
                                  value=30, step=1, key="tp1_dim")
        with c2:
            lb = st.number_input("Range min", value=float(lb_default),
                                 step=1.0, key="tp1_lb")
        with c3:
            ub = st.number_input("Range max", value=float(ub_default),
                                 step=1.0, key="tp1_ub")
        with c4:
            generate = st.button("🎲 Generate solution", use_container_width=True,
                                 key="tp1_gen")

        if generate or "tp1_solution" not in st.session_state:
            st.session_state["tp1_solution"] = np.random.uniform(lb, ub, size=int(dim))

        sol = st.session_state["tp1_solution"]

        colA, colB = st.columns([3, 1])
        with colA:
            st.text_input("Candidate solution example",
                          value=format_solution(sol),
                          disabled=True)
        with colB:
            # Display fitness if already computed
            fitness_val = st.session_state.get("tp1_fitness", "")
            st.text_input("Fitness", value=str(fitness_val),
                          disabled=True)

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown("**Function:**")
            st.markdown(f"`{fn_name}`")
            st.caption(info["description"])
            st.caption(f"Type: {info['type']}")
        with c2:
            st.latex(info["formula"])
        with c3:
            evaluate = st.button("✅ Evaluate solution", use_container_width=True,
                                 key="tp1_eval")

        if evaluate:
            sol = st.session_state["tp1_solution"]
            # If dimension changed, regenerate
            if len(sol) != int(dim):
                sol = np.random.uniform(lb, ub, size=int(dim))
                st.session_state["tp1_solution"] = sol
            val = fn(sol)
            st.session_state["tp1_fitness"] = round(val, 2)
            st.rerun()


# =============================================================================
# TP N°2 - Population Initialization (Part 2)
# =============================================================================
def page_tp2():

    st.title("TP - Metaheuristics")
    st.subheader("Part 2 \\ Population Initialization")

    # --- Function & problem setup ---
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fn_name = st.selectbox("Function", list(FUNCTIONS.keys()), key="tp2_fn")
    with col_b:
        dim = st.number_input("Dimension (D)", 1, 1000, 30, key="tp2_dim")
    fn, lb, ub, info = get_function(fn_name)
    with col_c:
        st.markdown("**Range**")
        st.markdown(f"`[{lb:g}, {ub:g}]`")

    # ---------------------------------------------------------------------
    # SECTION I: Population Initialization
    # ---------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### Population Initialization 🔗")

        col1, col2 = st.columns([3, 1])
        with col1:
            size = st.slider("Size", min_value=2, max_value=200, value=30, key="tp2_size")
        with col2:
            if st.button("🎲 Generate population", use_container_width=True, key="tp2_gen"):
                st.session_state["tp2_pop"] = np.random.uniform(
                    lb, ub, size=(int(size), int(dim))
                )

        uploaded = st.file_uploader("Drag and drop file here (CSV)",
                                    type=["csv"], key="tp2_csv")
        if uploaded is not None:
            pop = load_population_from_csv(uploaded)
            st.session_state["tp2_pop"] = pop
            st.success(f"Loaded population: shape = {pop.shape}")

    # ---------------------------------------------------------------------
    # SECTION II: Population Evaluation
    # ---------------------------------------------------------------------
    with st.container(border=True):
        col_eval, _ = st.columns([1, 4])
        with col_eval:
            eval_click = st.button("✅ Evaluate population", use_container_width=True,
                                   key="tp2_eval")

        if eval_click and "tp2_pop" in st.session_state:
            pop = st.session_state["tp2_pop"]
            fit = np.array([fn(x) for x in pop])
            st.session_state["tp2_fit"] = fit
            st.session_state["tp2_best_idx"] = int(np.argmin(fit))

        # --- Plots ---
        col_surf, col_scatter, col_stats = st.columns([1, 1, 1])
        with col_surf:
            st.caption(f"Function ({fn_name})")
            fig = surface_plot(fn, lb, ub, title="")
            st.pyplot(fig, use_container_width=True)

        with col_scatter:
            if "tp2_pop" in st.session_state:
                pop = st.session_state["tp2_pop"]
                best = None
                if "tp2_best_idx" in st.session_state:
                    best = pop[st.session_state["tp2_best_idx"]]
                fig = contour_scatter(fn, lb, ub,
                                      population=pop,
                                      best=best,
                                      title=f"Search History ({fn_name})")
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Generate a population to see the scatter plot.")

        with col_stats:
            if "tp2_fit" in st.session_state:
                fit = st.session_state["tp2_fit"]
                st.markdown("**Results**")
                st.info(f"**Best** — {np.min(fit):.2f}  \n"
                        f"**Worst** — {np.max(fit):.2f}")

    # ---------------------------------------------------------------------
    # SECTION III: Running Multiple Populations
    # ---------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### Running Multiple Populations")

        c1, c2 = st.columns([3, 1])
        with c1:
            n_runs = st.slider("Run", 1, 200, 30, key="tp2_runs")
        with c2:
            run_click = st.button("▶️ Evaluate", use_container_width=True, key="tp2_multi")

        if run_click:
            results = []
            all_best = []
            all_worst = []
            for r in range(int(n_runs)):
                pop = np.random.uniform(lb, ub, size=(int(size), int(dim)))
                fit = np.array([fn(x) for x in pop])
                results.append({"Run": r + 1,
                                "Best": np.min(fit),
                                "Worst": np.max(fit),
                                "Mean": np.mean(fit)})
                all_best.append(np.min(fit))
                all_worst.append(np.max(fit))
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, height=250)

            best_all = float(np.min(all_best))
            worst_all = float(np.max(all_worst))
            mean_all = float(np.mean([r["Mean"] for r in results]))
            std_all = float(np.std([r["Best"] for r in results], ddof=1)) \
                if len(results) > 1 else 0.0

            st.markdown("#### 📊 Global Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best", f"{best_all:.2f}")
            c2.metric("Worst", f"{worst_all:.2f}")
            c3.metric("Mean (AVG)", f"{mean_all:.2f}")
            c4.metric("STD", f"{std_all:.2f}")


# =============================================================================
# TP N°3 - PSO Part 1
# =============================================================================
def page_tp3():

    st.title("TP - Metaheuristics")
    st.subheader("Particle Swarm Optimization – Part 1")
    st.markdown("**Application of PSO for the selected function**")

    # --- Parameters ---
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            fn_name = st.selectbox("Function", list(FUNCTIONS.keys()),
                                   key="tp3_fn")
        with c2:
            dim = st.number_input("Dimension (D)", 2, 1000, 30, key="tp3_dim")
        with c3:
            pop_size = st.number_input("Population", 2, 500, 30, key="tp3_pop")
        with c4:
            metaheuristic = st.selectbox("Metaheuristic", ["PSO"], key="tp3_meta")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            max_iter = st.number_input("Max Iteration (T)", 1, 5000, 200, key="tp3_T")
        with c6:
            w = st.number_input("w", 0.0, 2.0, 0.3, step=0.1, key="tp3_w")
        with c7:
            c1_param = st.number_input("c1", 0.0, 4.0, 1.4, step=0.1, key="tp3_c1")
        with c8:
            c2_param = st.number_input("c2", 0.0, 4.0, 1.4, step=0.1, key="tp3_c2")

        col_btn = st.columns([1, 4])[0]
        with col_btn:
            run_click = st.button("▶️ Evaluate", use_container_width=True,
                                  key="tp3_run")

        fn, lb, ub, info = get_function(fn_name)

        if run_click:
            res = pso(fn, int(dim), lb, ub,
                      pop_size=int(pop_size), max_iter=int(max_iter),
                      w=float(w), c1=float(c1_param), c2=float(c2_param))
            st.session_state["tp3_result"] = res
            st.session_state["tp3_fn_name"] = fn_name
            st.session_state["tp3_range"] = (lb, ub)
            st.session_state["tp3_fn"] = fn

    # --- Display results ---
    if "tp3_result" in st.session_state:
        res = st.session_state["tp3_result"]
        fn_used = st.session_state["tp3_fn"]
        fn_name_used = st.session_state["tp3_fn_name"]
        lb_used, ub_used = st.session_state["tp3_range"]

        st.markdown("---")
        # --- Row 1: Surface | Init scatter | Final scatter | Stats ---
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.9])
        with col1:
            st.caption(f"Function ({fn_name_used})")
            st.pyplot(surface_plot(fn_used, lb_used, ub_used, title=""),
                      use_container_width=True)
        with col2:
            st.pyplot(contour_scatter(
                fn_used, lb_used, ub_used,
                population=res["initial_population"],
                best=res["initial_population"][np.argmin(
                    [fn_used(x) for x in res["initial_population"]])],
                title=f"Search History ({fn_name_used}), 1st Iteration",
            ), use_container_width=True)
        with col3:
            st.pyplot(contour_scatter(
                fn_used, lb_used, ub_used,
                population=res["position_history"][-1],
                best=res["best_solution"],
                trail=res["position_history"],
                title=f"Search History ({fn_name_used}), Final Iteration",
            ), use_container_width=True)
        with col4:
            st.markdown("**Initial population:**")
            st.markdown(f"**Best** — {res['initial_best']:.2f}, "
                        f"**Worst** — {res['initial_worst']:.2f}")
            st.markdown("**Final population:**")
            st.markdown(f"**Best** — {res['best_fitness']:.4f}")
            if res["stagnation_iter"] is not None:
                st.markdown(f"**Stagnation** — Iteration N°{res['stagnation_iter']}")
            else:
                st.markdown("**Stagnation** — Not detected")

        # --- Row 2: Convergence | Trajectory | Average fitness ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(line_plot(res["convergence_curve"],
                                title="Convergence Curve",
                                color="red"),
                      use_container_width=True)
        with col2:
            st.pyplot(line_plot(res["trajectory_first"],
                                title="Trajectory of 1st solution",
                                ylabel=r"$x_1^{(1)}$",
                                color="limegreen"),
                      use_container_width=True)
        with col3:
            st.pyplot(line_plot(res["average_fitness_curve"],
                                title="Average Fitness",
                                color="royalblue"),
                      use_container_width=True)


# =============================================================================
# TP N°4 - PSO Part 2 (Multiple Runs)
# =============================================================================
def page_tp4():

    st.title("TP - Metaheuristics")
    st.subheader("Running Multiple PSO Experiments")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fn_name = st.selectbox("Function", list(FUNCTIONS.keys()),
                                   key="tp4_fn")
        with c2:
            dim = st.number_input("Dimension (D)", 2, 1000, 30, key="tp4_dim")
        with c3:
            pop_size = st.number_input("Population", 2, 500, 30, key="tp4_pop")
        with c4:
            max_iter = st.number_input("Max Iteration (T)", 1, 5000, 200, key="tp4_T")

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            w = st.number_input("w", 0.0, 2.0, 0.5, step=0.1, key="tp4_w")
        with c6:
            c1_param = st.number_input("c1", 0.0, 4.0, 2.0, step=0.1, key="tp4_c1")
        with c7:
            c2_param = st.number_input("c2", 0.0, 4.0, 2.0, step=0.1, key="tp4_c2")
        with c8:
            n_runs = st.number_input("Run", 1, 200, 30, key="tp4_runs")

        run_click = st.button("▶️ Evaluate", key="tp4_run", use_container_width=False)

        fn, lb, ub, info = get_function(fn_name)

        if run_click:
            all_bests = []
            last_result = None
            progress = st.progress(0.0)
            for r in range(int(n_runs)):
                res = pso(fn, int(dim), lb, ub,
                          pop_size=int(pop_size), max_iter=int(max_iter),
                          w=float(w), c1=float(c1_param), c2=float(c2_param))
                all_bests.append(res["best_fitness"])
                last_result = res
                progress.progress((r + 1) / int(n_runs))
            progress.empty()

            st.session_state["tp4_result"] = last_result
            st.session_state["tp4_bests"] = np.array(all_bests)
            st.session_state["tp4_fn_name"] = fn_name
            st.session_state["tp4_range"] = (lb, ub)
            st.session_state["tp4_fn"] = fn
            st.session_state["tp4_params"] = {
                "Fonction": fn_name, "RUN": int(n_runs),
                "ITERATION": int(max_iter), "POPULATION": int(pop_size),
                "W": float(w), "C1": float(c1_param), "C2": float(c2_param),
            }

    if "tp4_result" in st.session_state:
        res = st.session_state["tp4_result"]
        bests = st.session_state["tp4_bests"]
        fn_used = st.session_state["tp4_fn"]
        fn_name_used = st.session_state["tp4_fn_name"]
        lb_used, ub_used = st.session_state["tp4_range"]
        params = st.session_state["tp4_params"]

        st.markdown("---")
        # --- Parameter table ---
        st.table(pd.DataFrame([params]))

        # --- Row 1: Surface | Final scatter | Stats ---
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.caption(f"Function ({fn_name_used})")
            st.pyplot(surface_plot(fn_used, lb_used, ub_used, title=""),
                      use_container_width=True)
        with col2:
            st.pyplot(contour_scatter(
                fn_used, lb_used, ub_used,
                population=res["position_history"][-1],
                best=res["best_solution"],
                trail=res["position_history"],
                title=f"Search History ({fn_name_used}), Final Iteration",
            ), use_container_width=True)
        with col3:
            best_all = float(np.min(bests))
            mean_all = float(np.mean(bests))
            std_all = float(np.std(bests, ddof=1)) if len(bests) > 1 else 0.0
            st.markdown(f"**Best** — {best_all:.2f}")
            st.markdown(f"**Mean (average error)** — {mean_all:.2f}")
            st.markdown(f"**STD** — {std_all:.2f}")

        # --- Row 2: Convergence | Trajectory | Average fitness ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(line_plot(res["convergence_curve"],
                                title="Convergence Curve", color="red"),
                      use_container_width=True)
        with col2:
            st.pyplot(line_plot(res["trajectory_first"],
                                title="Trajectory of 1st solution",
                                ylabel=r"$x_1^{(1)}$",
                                color="limegreen"),
                      use_container_width=True)
        with col3:
            st.pyplot(line_plot(res["average_fitness_curve"],
                                title="Average Fitness of population",
                                color="royalblue"),
                      use_container_width=True)


# =============================================================================
# TP N°7 - Genetic Algorithm
# =============================================================================
def page_tp7():

    st.title("TP - Metaheuristics")
    st.subheader("Genetic Algorithm")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fn_name = st.selectbox("Function", list(FUNCTIONS.keys()),
                                   key="tp7_fn")
        with c2:
            dim = st.number_input("Dimension (D)", 2, 1000, 30, key="tp7_dim")
        with c3:
            pop_size = st.number_input("Population", 2, 500, 30, key="tp7_pop")
        with c4:
            max_iter = st.number_input("Max Iteration (T)", 1, 5000, 200, key="tp7_T")

        c5, c6 = st.columns(2)
        with c5:
            pc = st.number_input("Crossover probability (pc)",
                                 0.0, 1.0, 0.8, step=0.05, key="tp7_pc")
        with c6:
            pm = st.number_input("Mutation probability (pm)",
                                 0.0, 1.0, 0.1, step=0.05, key="tp7_pm")

        run_click = st.button("▶️ Evaluate", key="tp7_run")

        fn, lb, ub, info = get_function(fn_name)

        if run_click:
            res = ga(fn, int(dim), lb, ub,
                     pop_size=int(pop_size), max_iter=int(max_iter),
                     pc=float(pc), pm=float(pm))
            st.session_state["tp7_result"] = res
            st.session_state["tp7_fn_name"] = fn_name
            st.session_state["tp7_range"] = (lb, ub)
            st.session_state["tp7_fn"] = fn

    if "tp7_result" in st.session_state:
        res = st.session_state["tp7_result"]
        fn_used = st.session_state["tp7_fn"]
        fn_name_used = st.session_state["tp7_fn_name"]
        lb_used, ub_used = st.session_state["tp7_range"]

        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.9])
        with col1:
            st.caption(f"Function ({fn_name_used})")
            st.pyplot(surface_plot(fn_used, lb_used, ub_used, title=""),
                      use_container_width=True)
        with col2:
            st.pyplot(contour_scatter(
                fn_used, lb_used, ub_used,
                population=res["initial_population"],
                best=res["initial_population"][np.argmin(
                    [fn_used(x) for x in res["initial_population"]])],
                title=f"Search History ({fn_name_used}), 1st Iteration",
            ), use_container_width=True)
        with col3:
            st.pyplot(contour_scatter(
                fn_used, lb_used, ub_used,
                population=res["position_history"][-1],
                best=res["best_solution"],
                trail=res["position_history"],
                title=f"Search History ({fn_name_used}), Final Iteration",
            ), use_container_width=True)
        with col4:
            st.markdown("**Initial population:**")
            st.markdown(f"**Best** — {res['initial_best']:.2f}, "
                        f"**Worst** — {res['initial_worst']:.2f}")
            st.markdown("**Final population:**")
            st.markdown(f"**Best** — {res['best_fitness']:.4f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.pyplot(line_plot(res["convergence_curve"],
                                title="Convergence Curve", color="red"),
                      use_container_width=True)
        with col2:
            st.pyplot(line_plot(res["trajectory_first"],
                                title="Trajectory of 1st solution",
                                ylabel=r"$x_1^{(1)}$",
                                color="limegreen"),
                      use_container_width=True)
        with col3:
            st.pyplot(line_plot(res["average_fitness_curve"],
                                title="Average Fitness",
                                color="royalblue"),
                      use_container_width=True)


# =============================================================================
# ROUTER
# =============================================================================
PAGES = {
    "TP N°1 – Optimization Problem Initialization": page_tp1,
    "TP N°2 – Population Initialization (Part 2)": page_tp2,
    "TP N°3 – PSO (Part 1)": page_tp3,
    "TP N°4 – PSO (Part 2 - Multiple Runs)": page_tp4,
    "TP N°7 – Genetic Algorithm": page_tp7,
}

PAGES[page]()

st.sidebar.markdown("---")