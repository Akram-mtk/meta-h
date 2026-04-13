import numpy as np
from functions import *


data = np.loadtxt("data/Population_F1-UM.csv", delimiter=";")





fitness, best_i, worst_i = evaluate_population(data, F1)


print(f"Best fitness:  {fitness[best_i]:.4f}")
print(f"Worst fitness: {fitness[worst_i]:.4f}")





# arr = np.array([-27.81, -71.96, -47.13, 54.63, -86.58, -96.77, 63.39, 75.60, -39.94, -45.13,
#                  90.77, 70.68, 36.61, 18.50, -46.11, -91.04, -34.74, 94.34, 61.98, -77.94,
#                 -78.75,  -3.11,  -2.81, 80.69, -76.95, 43.46,  3.65, -26.73, 49.26,  0.72])

# print(f"fitness = {F1(arr)}")

