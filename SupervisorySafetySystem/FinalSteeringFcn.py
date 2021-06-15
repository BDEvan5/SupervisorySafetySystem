from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state

def run_step(x, a):
    for i in range(10):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 


def single_model():
    plt.figure(1)
    plt.title("Position for 0.4 -> -0.4 angles")

    speed = 3
    ts = np.linspace(0, 0.5, 40)
    
    du = -0.4 
    d0 = 0.4

    a = np.array([du, 3])
    x = np.array([0, 0, 0, 3, d0])

    xs, ys = [0], [0]
    for i in range(5):
        x = run_step(x, a)
        xs.append(x[0])
        ys.append(x[1])

    plt.plot(xs, ys, '-+', linewidth=2)

    alphas = np.zeros_like(ts)
    for i, t in enumerate(ts):
        alpha = model(d0, du, t)
        alphas[i] = alpha
    t_transient = abs(d0-du)/3.2

    d_follow = (2*d0+du) / 3
    alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)
    ld_trans = speed * t_transient
    plt.plot(ld_trans*np.sin(alpha_trans), ld_trans*np.cos(alpha_trans), 'x', markersize=20)
    
    lds = speed * ts[ts<t_transient]
    plt.plot(lds*np.sin(alphas[ts<t_transient]), lds*np.cos(alphas[ts<t_transient]), '-x', linewidth=2)

    lds = speed * (ts[ts>=t_transient] - np.ones_like(ts[ts>=t_transient])*t_transient)
    x_trans = ld_trans*np.sin(alpha_trans)
    y_trans = ld_trans*np.cos(alpha_trans)
    xs = x_trans + lds * np.sin(alphas[ts>=t_transient])
    ys = y_trans + lds * np.cos(alphas[ts>=t_transient])

    plt.plot(xs, ys, '-x')

    print(f"--------------------------------")

    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# def redo_model():


def model(d0, du, t):
    speed = 3
    t_transient = abs(d0-du)/3.2
    sign = 1
    if du < d0:
        sign = -1

    if t < t_transient:
        d_follow = (3*d0 + 3.2*t * sign) / 3
        # d_follow = d0
        print(f"t: {t} -> dfollow: {d_follow}")
        alpha = np.arcsin(np.tan(d_follow)*speed*t/0.66)

    if t >= t_transient:
        d_follow = (2*d0+du) / 3
        alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)
        # theta = speed / 0.33 * np.tan(d_follow) * t_transient 

        alpha_prime = np.arcsin(np.tan(du)*speed*(t-t_transient)/0.66)

        alpha = alpha_trans + alpha_prime 

    return alpha

def run_calc_fcn():
    plt.figure(1)
    plt.title("Position for 0.4 -> -0.4 angles using model")

    speed = 3
    ts = np.linspace(0, 0.5, 40)
    
    du = -0.4 
    d0 = 0.4

    a = np.array([du, 3])
    x = np.array([0, 0, 0, 3, d0])

    xs, ys = [0], [0]
    for i in range(5):
        x = run_step(x, a)
        xs.append(x[0])
        ys.append(x[1])

    plt.plot(xs, ys, '-+', linewidth=1)

    # Start with my function
    xs, ys = np.zeros_like(ts), np.zeros_like(ts)
    alphas = np.zeros_like(ts)

    for i in range(len(ts)):
        # xs[i], ys[i] = run_model(d0, du, ts[i])
        alphas[i] = run_model(d0, du, ts[i])
        
    # plt.plot(xs, ys, '-+', linewidth=2)

    d_follow = (2*d0+du) / 3
    t_transient = abs(d0-du)/3.2
    alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)
    ld_trans = speed * t_transient
    plt.plot(ld_trans*np.sin(alpha_trans), ld_trans*np.cos(alpha_trans), 'x', markersize=20)
    
    lds = speed * ts[ts<t_transient]
    plt.plot(lds*np.sin(alphas[ts<t_transient]), lds*np.cos(alphas[ts<t_transient]), '-x', linewidth=2)

    lds = speed * (ts[ts>=t_transient] - np.ones_like(ts[ts>=t_transient])*t_transient)
    x_trans = ld_trans*np.sin(alpha_trans)
    y_trans = ld_trans*np.cos(alpha_trans)
    xs = x_trans + lds * np.sin(alphas[ts>=t_transient])
    ys = y_trans + lds * np.cos(alphas[ts>=t_transient])

    plt.plot(xs, ys, '-x')

    plt.show()

def run_model(d0, du, t):
    sv = 3.2 
    speed = 3
    L = 0.33

    t_transient = (du-d0)/sv
    sign = t_transient / abs(t_transient)
    t_transient = abs(t_transient)

    d_follow = (3*d0 + sv*t * sign) / 3
    alpha = np.arcsin(np.tan(d_follow)*speed*min(t, t_transient)/(L*2)) + \
        np.arcsin(np.tan(du)*speed*max(t-t_transient, 0)/(L*2))

    return alpha




single_model()
# run_calc_fcn()
