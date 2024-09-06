import gurobipy as gp
import numpy as np
import random

model = gp.Model('test')
print(model)

for i in range(0, 100):
    # var = np.random.normal(3.12e-5)
    # if var < 0:
    #     print(var)
    print(random.random())