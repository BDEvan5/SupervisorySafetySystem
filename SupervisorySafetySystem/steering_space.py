from matplotlib import pyplot as plt
import numpy as np
from numba import njit
from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state

def run_step(x, a):
    for i in range(10):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 

def plot_steering_angles():
    plt.figure(1)
    plt.title("Steering angles")

    deltas = np.linspace(-0.4, 0.4, 5)

    # lds = np.linspace(0, 1.5, 10)
    speed = 3
    ts = np.linspace(0, 0.5, 20)

    for d in deltas:
        a = np.array([d, 3])
        d0 = 0.4
        x = np.array([0, 0, 0, 3, d0])

        xs, ys = [0], [0]
        for i in range(5):
            x = run_step(x, a)
            xs.append(x[0])
            ys.append(x[1])

        plt.plot(xs, ys, '-+', linewidth=2)
        th = np.arctan(xs[-1]/ys[-1]) * 180/ np.pi
        print(f"D:{d:.4f} --> x: {xs[-1]:.4f} Y:{ys[-1]:.4f} --> th:{th:.4f}")

        t_change = abs(d - d0) / 3.2
        t_rest = 0.5-t_change
        d_follow = (d0+d)/2 
        alpha_o = np.arcsin(np.tan(d_follow)*3*t_change/0.66) * 180/ np.pi
        alpha_u = np.arcsin(np.tan(d)*3*t_rest/0.66)* 180/ np.pi

        model = alpha_o+alpha_u
        print(f"a0: {alpha_o:.4f} -> a_u: {alpha_u:.4f} -> total: {model:.4f} -> t_change: {t_change}")

        print(f"Actual: {th:.4f} --> Model: {model:.4f} --> Diff: {(th-model):.4f}")

        alphas = np.zeros_like(ts)
        for i, t in enumerate(ts):
            alphas[i] = calculate_alpha(d0, d, t)

        lds = speed * ts
        plt.plot(lds*np.sin(alphas), lds*np.cos(alphas), '--', linewidth=2)

        print(f"--------------------------------")

    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# @njit(cache=True)
def calculate_alpha(d0, du, t):
    speed = 3
    t_transient = abs(d0-du)/3.2
    sign = 1
    if du < d0:
        sign = -1

    if t < t_transient:
        d_follow = (2*d0 + 3.2*t * sign) / 2
        # d_follow = d0
        print(f"t: {t} -> dfollow: {d_follow}")
        alpha = np.arcsin(np.tan(d_follow)*speed*t/0.66)

        return alpha

    if t > t_transient:
        d_follow = (d0+du) / 2
        # d_follow = d0
        alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)

        alpha_prime = np.arcsin(np.tan(du)*speed*(t-t_transient)/0.66)

        theta = speed / 0.33 * np.tan(d_follow) * t_transient

        alpha = alpha_trans + alpha_prime 

        return alpha

def plot_steering_angles_fine():
    plt.figure(1)
    plt.title("Steering angles")

    deltas = np.linspace(-0.4, 0.4, 7)

    lds = np.linspace(0, 1.5, 20)

    for d in deltas:
        a = np.array([d, 3])
        x = np.array([0, 0, 0, 3, 0.4])

        xs, ys = [0], [0]
        for i in range(500):
            u = control_system(x, a, 7, 0.4, 8, 3.2)
            x = update_kinematic_state(x, u, 0.001, 0.33, 0.4, 7)
            # x = run_step(x, a)
            xs.append(x[0])
            ys.append(x[1])

        plt.plot(xs, ys, '-+')

        alphas = np.arcsin(np.tan(d)*lds/0.66)
        plt.plot(lds*np.sin(alphas), lds*np.cos(alphas), '-x')

    plt.show()

def single_model():
    plt.figure(1)
    plt.title("Steering angles")

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
        t_transient = abs(d0-du)/3.2
        sign = 1
        if du < d0:
            sign = -1

        if t < t_transient:
            d_follow = (3*d0 + 3.2*t * sign) / 3
            # d_follow = d0
            print(f"t: {t} -> dfollow: {d_follow}")
            alpha = np.arcsin(np.tan(d_follow)*speed*t/0.66)

        if t > t_transient:
            d_follow = (2*d0+du) / 3
            alpha_trans = np.arcsin(np.tan(d_follow)*speed*t_transient/0.66)
            theta = speed / 0.33 * np.tan(d_follow) * t_transient 

            alpha_prime = np.arcsin(np.tan(du)*speed*(t-t_transient)/0.66)

            alpha = alpha_trans + alpha_prime 

        alphas[i] = alpha

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

def sinlge_function():
    plt.figure(1)
    plt.title("Steering angles")

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
    lds = np.zeros_like(ts)
    for i, t in enumerate(ts):
        t_transient = (du-d0)/3.2
        sign = t_transient / abs(t_transient)
        t_transient = abs(t_transient)

        d_follow = (3*d0 + 3.2*t * sign) / 3
        alpha = np.arcsin(np.tan(d_follow)*speed*min(t, t_transient)/0.66) + \
            np.arcsin(np.tan(du)*speed*max(t-t_transient, 0)/0.66)

        alphas[i] = alpha

        ld = 0.33*np.tan(d_follow)*2*alpha 


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

# def project_point(x, a):



# def inverse_step()

def inverse_model(x, alpha, ld):
    """
    This inverse model must take a state and an alpha and then return the constant steering angle to get there.

    Args:
        x:current states
        alpha: desired angle 
        ld: distance to that point.
    """
    pass

# plot_steering_angles()
# plot_steering_angles_fine()
single_model()
    
sinlge_function()





