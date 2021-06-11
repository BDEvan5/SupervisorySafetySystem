from matplotlib import pyplot as plt
import numpy as np

from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state

def run_step(x, a):
    for i in range(10):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 

def plot_steering_angles():
    plt.figure(1)
    plt.title("Steering angles")

    deltas = np.linspace(-0.4, 0.4, 7)

    lds = np.linspace(0, 1.5, 20)

    for d in deltas:
        a = np.array([d, 3])
        x = np.array([0, 0, 0, 3, 0])

        xs, ys = [0], [0]
        for i in range(5):
            x = run_step(x, a)
            xs.append(x[0])
            ys.append(x[1])

        plt.plot(xs, ys, '-+')

        alphas = np.arcsin(np.tan(d)*lds/0.8)
        plt.plot(lds*np.sin(alphas), lds*np.cos(alphas), '-x')

    plt.show()

plot_steering_angles()

    






