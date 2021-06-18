from sympy import *
import numpy as np

V = 3 
L = 0.33 
sv = 3.2

# d0, du, t, t0 = symbols('d0 du t t0')
d0 = symbols('d0')
du = symbols('du')
t = symbols('t')
t0 = symbols('t0')
x_p = symbols('x_p')

# x_p = V**2 * t0**2 * np.tan((3*d0 + du*sv*t0)/3) / (2*L) #
# x_p = V**2 * t0**2 * np.tan(du) / (2*L) #

eq1 = Eq((V**2 * t0**2 * np.tan(du) )/ (2*L), x_p)

s = solve(eq1, t0)

init_printing(use_unicode=True)

print(x_p)
print(s)

