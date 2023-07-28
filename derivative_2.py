import numpy as np
from sympy import Symbol, log

x = Symbol('x')

def derivative(f, x_val):   
    fprime = f.diff()
    print(fprime)
    res = fprime.subs(x, x_val)
    return round(float(res), 3)

f = log(x**3)
result = derivative(f, 1.0)
print(result)
