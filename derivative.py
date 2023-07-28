import numpy as np
from sympy import Symbol

x1 = Symbol('x1')
x2 = Symbol('x2')
y = 3*(x1**3)*(x2**3) - (x1**(1/2))*(x2**2)
yp = y.diff(x1)
yp_val = yp.subs(x1,1).subs(x2,1)
print(yp_val)

