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

def steering_model_clean(d0, du, t):
    speed = 0.5
    L = 0.33

    t_transient = (du-d0)/3.2
    sign = t_transient / abs(t_transient)
    t_transient = abs(t_transient)

    ld_trans = speed * min(t, t_transient)
    d_follow = (3*d0 + 3.2*min(t, t_transient) * sign) / 3
    alpha_trans = np.arcsin(np.tan(d_follow)*ld_trans/(2*L))

    ld_prime = speed * max(t-t_transient, 0)
    alpha_prime = np.arcsin(np.tan(du)*ld_prime/(2*L))
    # reason for adding them is the old alpha is the base theta for the next step
    alpha_ss = alpha_trans + alpha_prime 

    x = ld_trans * np.sin(alpha_trans) + ld_prime*np.sin(alpha_ss)
    y = ld_trans * np.cos(alpha_trans) + ld_prime*np.cos(alpha_ss)

    return x, y


def run_calc_fcn():
    plt.figure(1)
    plt.title("Position for 0.4 -> -0.4 angles using model")

    ts = np.linspace(0, 0.5, 40)
    du = -0.4 
    d0 = 0.4

    a = np.array([du, 0.5])
    x = np.array([0, 0, 0, 0.5, d0])
    xs, ys = [0], [0]
    for i in range(5):
        x = run_step(x, a)
        xs.append(x[0])
        ys.append(x[1])

    plt.plot(xs, ys, '-+', linewidth=1)

    # Start with my function
    xs, ys = np.zeros_like(ts), np.zeros_like(ts)

    for i in range(len(ts)):
        xs[i], ys[i] = steering_model_clean(d0, du, ts[i])
        
    plt.plot(xs, ys, '-x')
    x, y = steering_model_clean(d0, du, abs(d0-du)/3.2)
    plt.plot(x, y, 'x', markersize=20)
    # plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def inverse_model(x, x_p):
    """
    Calculates the required steering input for a vehicle to move from the current state to the following state.

    Args:
        x: state of dim=5
        x_p: next state of dim=5

    Returns
        du: steering control input 
    """



run_calc_fcn()
