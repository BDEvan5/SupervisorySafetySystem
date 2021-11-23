import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from SupervisorySafetySystem.Simulator.Dynamics import update_complex_state
from SupervisorySafetySystem.KernelTests.GeneralTestTrain import load_conf


def build_discrim_dynamics(phis, qs, velocity, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    dynamics = np.zeros((len(phis), len(qs), 9, 3), dtype=np.int)
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
                state = np.array([0, 0, p, velocity, 0])
                action = np.array([m, velocity])
                new_state = update_complex_state(state, action, time)
                dx, dy, phi = new_state[0], new_state[1], new_state[2]

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi < -np.pi:
                    phi = phi + 2*np.pi

                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)                

                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, 8, 2] = min(max(0, new_k), len(phis)-1)

                dynamics[i, j, 0, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, 0, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, 1, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, 1, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, 2, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, 2, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, 3, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, 3, 1] = int(round((dy -h) * resolution))
                
                dynamics[i, j, 0+4, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, 0+4, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, 1+4, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, 1+4, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, 2+4, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, 2+4, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, 3+4, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, 3+4, 1] = int(round((dy -h) * resolution))
                
                dynamics[i, j, 8, 0] = int(round((dx) * resolution))
                dynamics[i, j, 8, 1] = int(round((dy) * resolution))

    return dynamics

conf = load_conf("track_kernel")
phis = np.linspace(-np.pi, np.pi, conf.n_phi)
qs = np.linspace(-conf.max_steer, conf.max_steer, conf.n_modes)
velocity = 2
time = conf.kernel_time_step
dynamics = build_discrim_dynamics(phis, qs, velocity, time, conf)

plt.figure(1)
phi = (conf.n_phi-1)//2
# phi = 30
# phi = 10
slice = dynamics[phi, :, :, :]

arrow_length = 3
plt.plot(0, 0, 'x', markersize=20, color='black')
angle = (phi - (conf.n_phi-1)//2) * np.pi * 2 / (conf.n_phi-1)
plt.arrow(0, 0, arrow_length*np.sin(angle), arrow_length*np.cos(angle), color='blue', head_width=0.4)
for i in range(conf.n_modes):
    n = 0
    x = slice[i, 8, 0]
    y = slice[i, 8, 1]
    plt.plot(x, y, 'x', color='green')
    angle = (slice[i, n, 2] - (conf.n_phi-1)//2) * np.pi * 2 / (conf.n_phi-1)
    plt.arrow(x, y, arrow_length*np.sin(angle), arrow_length*np.cos(angle), color='green', head_width=0.4)
    for n in range(8):
        x = slice[i, n, 0]
        y = slice[i, n, 1]
        plt.plot(x, y, 'x', color='red')

        angle = (slice[i, n, 2] - (conf.n_phi-1)//2) * np.pi * 2 / (conf.n_phi-1)
        plt.arrow(x, y, arrow_length*np.sin(angle), arrow_length*np.cos(angle), color='red', head_width=0.4)

        print(f"i: {i}, n:{n} --> X: {x}, Y: {y}, Theta: {angle}")

plt.gca().set_aspect('equal', adjustable='box')
plt.show()