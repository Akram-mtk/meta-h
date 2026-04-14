import streamlit as st
import numpy as np
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
    st.markdown("")  # spacing
    st.markdown("")  # spacing
    generate_btn = st.button("Generate solution", use_container_width=True)

# Generate or keep solution in session state
if "solution" not in st.session_state:
    st.session_state.solution = None
    st.session_state.fitness = None

if generate_btn:
    st.session_state.solution = np.random.uniform(lb, ub, int(dim))
    st.session_state.fitness = None  # reset fitness when new solution is generated

# Display candidate solution and fitness
col_sol, col_fit = st.columns([4, 1])

with col_sol:
    st.markdown("Candidate solution example")
    if st.session_state.solution is not None:
        # Format as pipe-separated string like the teacher
        sol_str = " | ".join([f"{v:.2f}" for v in st.session_state.solution])
        st.text_area("", value=sol_str, height=68, disabled=True, label_visibility="collapsed")
    else:
        st.text_area("", value="", height=68, disabled=True, label_visibility="collapsed")

with col_fit:
    st.markdown("Fitness")
    if st.session_state.fitness is not None:
        st.text_input("", value=f"{st.session_state.fitness:.2f}", disabled=True,
                       label_visibility="collapsed")
    else:
        st.text_input("", value="", disabled=True, label_visibility="collapsed")

st.markdown("")

# --- Function row ---
col_func, col_formula, col_eval = st.columns([1, 3, 1])

with col_func:
    st.markdown("**Function:**")
    selected_func = st.selectbox("", list(FUNCTIONS.keys()), label_visibility="collapsed")

with col_formula:
    cfg = FUNCTIONS[selected_func]
    st.latex(cfg["latex"])

with col_eval:
    st.markdown("")
    st.markdown("")
    evaluate_btn = st.button("Evaluate solution", use_container_width=True)

# When function is selected, update the range to match
if selected_func:
    cfg = FUNCTIONS[selected_func]
    # Auto-update range when function changes (only if user hasn't manually changed it)

# Evaluate
if evaluate_btn:
    if st.session_state.solution is not None:
        cfg = FUNCTIONS[selected_func]
        fitness = cfg["func"](st.session_state.solution)
        st.session_state.fitness = fitness
        st.rerun()
    else:
        st.warning("Please generate a solution first.")

st.divider()