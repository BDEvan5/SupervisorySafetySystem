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

def steering_model_clean(d0, du, t, speed=3):
    L = 0.33

    t_transient = (du-d0)/3.2
    sign = t_transient / abs(t_transient)
    t_transient = abs(t_transient)

    ld_trans = speed * min(t, t_transient)
    d_follow = (3*d0 + 3.2*min(t, t_transient) * sign) / 3
    alpha_trans = np.arcsin(np.tan(d_follow)*ld_trans/(2*L))

    ld_prime = speed * max(t-t_transient, 0)
    alpha_prime = np.arcsin(np.tan(du)*ld_prime/(2*L))
    alpha_ss = alpha_trans + alpha_prime
    
    x = ld_trans * np.sin(alpha_trans) + ld_prime*np.sin(alpha_ss)
    y = ld_trans * np.cos(alpha_trans) + ld_prime*np.cos(alpha_ss)

    return x, y


def run_calc_fcn():
    plt.figure(1)
    plt.title("Position for 0.4 -> -0.4 angles using model")

    ts = np.linspace(0, 0.5, 40)
    du = -0.3 
    d0 = 0.3

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

    print(f"X: {xs[-1]:.4f} Y: {ys[-1]:.4f} for d0: {d0} --> du:{du}")
        
    plt.plot(xs, ys, '-x')
    x, y = steering_model_clean(d0, du, abs(d0-du)/3.2)
    plt.plot(x, y, 'x', markersize=20)
    # plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

def get_distance(x, y, pt):
    return np.sqrt((x-pt[0])**2 + (y-pt[1])**2)

def inverse_model(x, x_p):
    """
    Calculates the required steering input for a vehicle to move from the current state to the following state.

    Args:
        x: state of dim=5
        x_p: next state of dim=2 (x, y)

    Returns
        du: steering control input 
    """
    max_steer = 0.4 
    n_steer = 50 

    # t = get_distance(0, 0, x_p) / x[3]
    t = 0.5
    print(f"Time as: {t}")

    d0 = x[4]
    if x_p[0] > x[0]:
        dus = np.linspace(0, max_steer, n_steer)
    else:
        dus = np.linspace(-max_steer, 0, n_steer)

    distances = np.zeros(n_steer)
    for i in range(n_steer):
        px, py = steering_model_clean(d0, dus[i], t)
        dis = get_distance(px, py, x_p[0:2])
        distances[i] = dis 

    ind = np.argmin(distances)

    return dus[ind]


def run_inv_model():
    x = np.array([0, 0, 0, 0.5, 0.3])
    x_p = np.array([-0.0079, 0.2497])

    du = inverse_model(x, x_p)
    print(f"Du: {du}")

def run_comparison():
    x = np.array([0, 0, 0, 3, 0.3])
    # x_p = np.array([-0.0079, 0.2497])

    du_set = -0.3
    px, py = steering_model_clean(x[4], du_set, 0.5, x[3])
    print(f"X: {px:.4f} Y: {py:.4f} for d0: {x[4]} --> du set:{du_set}")

    x_p = np.array([px, py])
    du_guess = inverse_model(x, x_p)
    print(f"Du guess: {du_guess}")
    px, py = steering_model_clean(x[4], du_guess, 0.5, x[3])
    print(f"Guess: X: {px:.4f} Y:{py:.4f} for distance: {get_distance(px, py, x_p):.4f}")


run_comparison()
# run_inv_model()
# run_calc_fcn()

