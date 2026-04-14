import numpy as np

# F1 — Sphere function
def F1(x):
    return np.sum(x**2)

# F2 — Sum + Product of abs
def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

# F5 — Rosenbrock
def F5(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# F7 — Quartic with noise
def F7(x):
    n = len(x)
    return np.sum((np.arange(1, n+1)) * x**4) + np.random.random()

# F8 — Schwefel
def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

# F9 — Rastrigin
def F9(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

# F11 — Griewank
def F11(x):
    n = len(x)
    i = np.arange(1, n+1)
    return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(i)))


# Function configs: name, function, default range, formula in LaTeX
FUNCTIONS = {
    "F1-UM": {
        "func": F1,
        "lb": -100, "ub": 100,
        "latex": r"f(x) = \sum_{i=1}^{D} x_i^2",
    },
    "F2-UM": {
        "func": F2,
        "lb": -10, "ub": 10,
        "latex": r"f(x) = \sum_{i=1}^{D} |x_i| + \prod_{i=1}^{D} |x_i|",
    },
    "F5-UM": {
        "func": F5,
        "lb": -30, "ub": 30,
        "latex": r"f(x) = \sum_{i=1}^{D-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]",
    },
    "F7-UM": {
        "func": F7,
        "lb": -128, "ub": 128,
        "latex": r"f(x) = \sum_{i=1}^{D} i \cdot x_i^4 + \text{random}(0,1)",
    },
    "F8-MM": {
        "func": F8,
        "lb": -500, "ub": 500,
        "latex": r"f(x) = \sum_{i=1}^{D} -x_i \sin\left(\sqrt{|x_i|}\right)",
    },
    "F9-MM": {
        "func": F9,
        "lb": -5.12, "ub": 5.12,
        "latex": r"f(x) = \sum_{i=1}^{D} \left[x_i^2 - 10\cos(2\pi x_i) + 10\right]",
    },
    "F11-MM": {
        "func": F11,
        "lb": -600, "ub": 600,
        "latex": r"f(x) = 1 + \frac{1}{4000}\sum_{i=1}^{D} x_i^2 - \prod_{i=1}^{D} \cos\left(\frac{x_i}{\sqrt{i}}\right)",
    },
}



def evaluate_population(pop, fitness_func):
    """Evaluate all solutions, return fitness values, best and worst"""
    fitness = np.array([fitness_func(sol) for sol in pop])
    best_idx = np.argmin(fitness)   # minimization
    worst_idx = np.argmax(fitness)
    return fitness, best_idx, worst_idx





F1f = np.array([-27.81, -71.96, -47.13, 54.63, -86.58, -96.77, 63.39, 75.60, 
               -39.94, -45.13, 90.77, 70.68, 36.61, 18.50, -46.11, -91.04, 
               -34.74, 94.34, 61.98, -77.94, -78.75, -3.11, -2.81, 80.69, 
               -76.95, 43.46, 3.65, -26.73, 49.26, 0.72])

# F2 Array
F2f = np.array([9.79, -1.60, 6.62, 6.07, 8.16, -1.29, 3.97, 7.41, 
               1.13, -7.32, 7.15, 3.59, 5.92, -9.07, 2.34, -2.47, 
               -7.22, 4.24, -7.75, -9.37, 7.08, 5.02, -2.03, -4.33, 
               1.50, -2.29, 7.32, 9.72, -9.67, -2.52])

# F5 Array
F5f = np.array([-23.85, -19.42, 6.20, -9.92, -1.11, 6.19, -3.25, 10.30, 
               22.39, -10.27, 8.92, 28.76, -10.92, 28.92, 12.16, -23.71, 
               13.74, -12.67, -23.86, 19.71, -25.63, 28.86, -28.97, 11.99, 
               6.37, -8.29, -6.86, 1.50, -25.89, 19.02])

# F7 Array
F7f = np.array([15.30, -124.59, 15.30, 76.61, -24.73, 60.58, -37.78, 89.80, 
               -77.46, 71.79, 1.68, 54.56, -9.37, -48.23, 10.54, -46.11, 
               -94.84, 9.99, -10.71, 19.30, -46.78, 20.22, -109.43, -66.11, 
               103.04, 124.89, -33.52, 43.97, -52.42, -71.94])

# F8 Array
F8f = np.array([294.94, 93.76, 77.03, -44.49, 338.08, 252.79, 318.38, 54.94, 
               428.00, 466.38, 306.97, 344.46, 469.53, 251.11, -198.22, 
               182.60, 362.80, 322.47, 377.28, 237.63, 131.96, -58.09, 
               -240.77, -484.17, -464.34, -152.18, -38.75, 369.36, -135.30, -249.50])

# F9 Array
F9f = np.array([-1.20, -1.63, 2.42, -4.95, -2.28, 1.16, 0.02, 0.71, 
               0.80, 4.00, 4.92, -1.61, -0.77, 2.36, 4.36, -5.09, 
               -1.42, 0.21, -1.96, 3.47, 3.08, -3.75, 3.71, 3.97, 
               3.80, -3.91, -0.09, 4.51, -1.06, 0.68])

# F11 Array
F11f = np.array([377.01, -369.41, 81.54, -546.71, -60.41, -338.54, -561.90, 
                -268.59, -21.93, 118.09, 150.80, -507.26, 25.70, -540.82, 
                146.39, 243.18, -107.89, 242.99, -281.18, 195.16, 301.57, 
                78.46, -447.01, -492.81, -488.54, -24.41, -22.75, -324.18, 
                437.47, -595.42])