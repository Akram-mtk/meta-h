# TP – Metaheuristics (Streamlit)

Reproduction of the teacher's interface for the **MÉTA** module (Master 2 SII – USTHB).

## Structure

```
meta_tp_app/
├── app.py           # Streamlit entry point with sidebar navigation
├── benchmarks.py    # Benchmark functions F1, F2, F5, F7, F8, F9, F11
├── pso.py           # Particle Swarm Optimization
├── ga.py            # Real-coded Genetic Algorithm
├── plotting.py      # Matplotlib helpers (surface, contour, line plots)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

## TPs implemented

| TP | Title | What it does |
|----|----|----|
| **TP N°1** | Optimization Problem Initialization | Generate one candidate solution, evaluate it on a benchmark function. |
| **TP N°2** | Population Initialization Part 2 | Generate / load a population (CSV), evaluate it, scatter-plot + best/worst, multi-run statistics (Best/Worst/Mean/STD). |
| **TP N°3** | PSO Part 1 | Single PSO run with convergence curve, search-history scatter (1st vs. final iteration), trajectory of 1st solution, average fitness, stagnation detection. |
| **TP N°4** | PSO Part 2 – Multiple Runs | Run PSO N times, aggregate statistics (Best / Mean / STD), display the last run's curves + scatter, and a parameter summary table. |
| **TP N°7** | Genetic Algorithm | Real-coded GA (tournament selection, arithmetic crossover, uniform mutation, elitist replacement) with the same visual layout as PSO. |

## CSV population format (TP N°2)

The app accepts the teacher's format: each line is one solution, values
separated by `;` (semicolons) or `,` (commas). The `Population_*.csv` files
you have work directly.

## Benchmark functions

| Name | Range | Min | Type |
|----|----|----|----|
| F1-UM | [-100, 100] | 0 | Unimodal – Sphere |
| F2-UM | [-10, 10] | 0 | Unimodal – Schwefel 2.22 |
| F5-UM | [-30, 30] | 0 | Unimodal – Rosenbrock |
| F7-UM | [-128, 128] | 0 | Unimodal – Noisy quartic |
| F8-MM | [-500, 500] | -418.9829·D | Multimodal – Schwefel |
| F9-MM | [-5.12, 5.12] | 0 | Multimodal – Rastrigin |
| F11-MM | [-600, 600] | 0 | Multimodal – Griewank |
