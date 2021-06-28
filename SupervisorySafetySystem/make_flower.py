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

    if d0 == du:
        t_transient, sign = 0, 0 #
    else:
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



def plot_steering_angles_fine():
    plt.figure(1)
    plt.title("Steering angles")

    deltas = np.linspace(-0.4, 0.4, 7)

    for d in deltas:
        a = np.array([d, 3])
        d0 = 0.0
        x = np.array([0, 0, 0, 3, d0])

        xs, ys = [0], [0]
        mxs, mys = [0], [0]
        for i in range(100):
            u = control_system(x, a, 7, 0.4, 8, 3.2)
            x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)
            # x = run_step(x, a)
            xs.append(x[0])
            ys.append(x[1])

            mx, my = steering_model_clean(d0, d, 0.01*i)
            mxs.append(mx)  
            mys.append(my)

        plt.plot(xs, ys,)
        plt.plot(mxs, mys, '-+', linewidth=2)




    plt.show()


plot_steering_angles_fine()


