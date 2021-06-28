import sympy as sy 
import mpmath as m 

sy.init_printing(use_unicode=True)

def basic_delta():
    du, t = sy.symbols('du t')
    v, L, d0 = sy.symbols('v L d0')
    x_p = sy.symbols('x_p')

    eq1 = v**2 * t**2 * sy.tan((2*d0 + du)/3) / (2*L) - x_p


    a = sy.solve(eq1, du)
    print(a)

def complex_delta():
    du, t = sy.symbols('du t') # to calculate
    t1, a_t, a_ss, ld_t, ld, d_t = sy.symbols('t1 a_t a_ss ld_t ld d_t') # intermediate
    d0, x, y = sy.symbols('d0 x y') # given 
    sv, L, v = sy.symbols('sv L v') # params

    eq1 = (d0-du)/sv - t1 
    eq2 = ld_t*sy.sin(a_t) + ld * sy.sin(a_ss) - x 
    eq3 = ld_t*sy.cos(a_t) + ld * sy.cos(a_ss) - y 
    eq4 = sy.asin(sy.tan(d_t)*ld_t/(2*L)) - a_t
    eq5 = sy.asin(sy.tan(du)*ld/(2*L)) + a_t - a_ss
    eq6 = (2*d0 + du)/3 - d_t 
    eq7 = t1*v - ld_t
    eq8 = (t-t1) * v - ld

    a = sy.solve_poly_system([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8], du, t)
    print(a)

    sy.pprint(a)

def less_complex_delta():
    du, t = sy.symbols('du t') # to calculate
    t1, a_t, a_ss, ld_t, ld, d_t = sy.symbols('t1 a_t a_ss ld_t ld d_t') # intermediate
    d0, x, y = sy.symbols('d0 x y') # given 
    sv, L, v = sy.symbols('sv L v') # params

    eq1 = (d0-du)/sv - t1 
    eq2 = t1*v *sy.sin(a_t) + (t-t1) * v * sy.sin(a_ss) - x 
    eq3 = t1*v *sy.cos(a_t) + (t-t1) * v * sy.cos(a_ss) - y 
    eq4 = sy.asin(sy.tan((2*d0 + du)/3)*t1*v /(2*L)) - a_t
    eq5 = sy.asin(sy.tan(du)*(t-t1) * v/(2*L)) + a_t - a_ss


    a = sy.solve([eq1, eq2, eq3, eq4, eq5], du, t, t1, a_t, a_ss)
    print(a)

    sy.pprint(a)

def even_less_complex_delta():
    du, t = sy.symbols('du t') # to calculate
    t1, a_t, a_ss, ld_t, ld, d_t = sy.symbols('t1 a_t a_ss ld_t ld d_t') # intermediate
    d0, x, y = sy.symbols('d0 x y') # given 
    sv, L, v = sy.symbols('sv L v') # params

    eq1 = t1*v *sy.tan((2*d0 + du)/3)*t1*v /(2*L) + (t-t1) * v * (sy.tan((2*d0 + du)/3)*t1*v /(2*L) + sy.tan(du)*(t-t1) * v/(2*L)) - x

    eq2 = t1*v * sy.cos(sy.asin(sy.tan((2*d0 + du)/3)*t1*v /(2*L))) + (t-t1) * v * sy.cos(sy.asin((sy.tan((2*d0 + du)/3)*t1*v /(2*L) + sy.tan(du)*(t-t1) * v/(2*L)))) - y


    # eq1 = (d0-du)/sv - t1 
    # eq2 = t1*v *sy.sin(a_t) + (t-t1) * v * sy.sin(a_ss) - x 
    # eq3 = t1*v *sy.cos(a_t) + (t-t1) * v * sy.cos(a_ss) - y 
    # eq4 = sy.asin(sy.tan((2*d0 + du)/3)*t1*v /(2*L)) - a_t
    # eq5 = sy.asin(sy.tan(du)*(t-t1) * v/(2*L)) + a_t - a_ss


    # a = sy.solve(eq1, t)
    # print(a)
    # a = sy.solve(eq2, du)
    # print(a)

    a = sy.solve([eq1, eq2], du, t)
    print(a)

    

    sy.pprint(a)

def single_order():
    du, t = sy.symbols('du t') # to calculate
    d0, x, y = sy.symbols('d0 x y') # given 
    sv, L, v = sy.symbols('sv L v') # params

    eq1 = (v * t)**2 * sy.tan(du) / (2*L)-x
    eq2 = v*t* sy.cos(sy.asin(sy.tan(du)*v*t/(2*L)))-y

    a = sy.solve([eq1, eq2], du, t)
    print(a)


# basic_delta()
# less_complex_delta()
even_less_complex_delta()
# complex_delta()
# single_order()
