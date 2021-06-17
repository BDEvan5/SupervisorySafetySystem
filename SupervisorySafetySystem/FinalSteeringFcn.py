from operator import mod
from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state

def run_step(x, a):
    for i in range(10):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 




def steering_model(d0, du, t):
    speed = 3
    t_transient = (du-d0)/3.2
    sign = t_transient / abs(t_transient)
    t_transient = abs(t_transient)

    if t < t_transient:
        d_follow = (3*d0 + 3.2*t * sign) / 3
        alpha = np.arcsin(np.tan(d_follow)*speed*t/0.66)

        ld = speed * t 
        x = ld * np.sin(alpha)
        y = ld * np.cos(alpha)

    if t >= t_transient:
        d_follow = (2*d0+du) / 3
        alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)
        alpha_prime = np.arcsin(np.tan(du)*speed*(t-t_transient)/0.66)
        alpha = alpha_trans + alpha_prime 

        ld_trans = speed * t_transient
        ld_prime = speed * (t-t_transient)
        x = ld_trans * np.sin(alpha_trans) + ld_prime*np.sin(alpha)
        y = ld_trans * np.cos(alpha_trans) + ld_prime*np.cos(alpha)

    return x, y


def run_calc_fcn():
    plt.figure(1)
    plt.title("Position for 0.4 -> -0.4 angles using model")

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

    for i in range(len(ts)):
        xs[i], ys[i] = steering_model(d0, du, ts[i])
        
    plt.plot(xs, ys, '-x')
    x, y = steering_model(d0, du, abs(d0-du)/3.2)
    plt.plot(x, y, 'x', markersize=20)
    # plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


run_calc_fcn()
