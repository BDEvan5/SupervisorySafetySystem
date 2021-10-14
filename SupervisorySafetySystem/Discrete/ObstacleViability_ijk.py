import numpy as np
from numba import njit, jit
from matplotlib import pyplot as plt, rc_context
import timeit

from numpy.core.defchararray import mod



class ViabilityKernel:
    def __init__(self, width=1, length=2):
        self.resolution = 200
        self.t_step = 0.2
        # self.n_steps = 10
        self.velocity = 2
        self.n_phi = 21
        self.phi_range = np.pi
        self.half_block = 1 / (2*self.resolution)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = 9

        self.n_x = self.resolution * width +1
        self.n_y = self.resolution * length +1
        self.x_offset = -0.25 
        self.y_offset = -1.5
        self.xs = np.linspace(self.x_offset, self.x_offset + width, self.n_x)
        self.ys = np.linspace(self.y_offset, self.y_offset + length, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.resolution)
        self.set_track_constraints()

    # config functions
    def build_qs(self):
        ds = np.linspace(-0.4, 0.4, self.n_modes)
        self.qs = self.velocity / 0.33 * np.tan(ds)

    def set_track_constraints(self):
        obs_size = 0.5 
        obs_offset = int(obs_size*self.resolution)
        x_start = int(-self.x_offset*self.resolution)
        y_start = int(-self.y_offset*self.resolution)
        self.kernel[x_start:x_start+obs_offset, y_start:y_start + obs_offset, :] = 1

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

            plt.figure(2)
            plt.title(f"Kernel after loop: {z}")
            img = self.kernel[:, :, 10].T - self.previous_kernel[:, :, 10].T
            plt.imshow(img, origin='lower')
            plt.pause(0.0001)

            self.view_kernel(0, False)

        np.save("SupervisorySafetySystem/Discrete/ObsKernal_ijk.npy", self.kernel)
        print(f"Saved kernel to file")

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/ObsKernal_ijk.npy")

    def view_kernel(self, phi, show=True):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        mode = 4
        img = self.kernel[:, :, phi_ind].T 
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, -1]


            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)
        if show:
            plt.show()


# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, resolution):
    # add 5 sample points
    n_pts = 5
    dynamics = np.zeros((len(phis), len(qs), n_pts, 3), dtype=np.int)
    phi_range = np.pi
    n_steps = 1
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            for t in range(n_pts):
                t_step = time * (t+1)  / n_pts
                phi = p + m * t_step * n_steps # phi must be at end
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 2] = min(max(0, new_k), len(phis)-1)
                dx = np.sin(phi) * velocity * t_step
                dynamics[i, j, t, 0] = int(round(dx * resolution))
                dy = np.cos(phi) * velocity * t_step
                dynamics[i, j, t, 1] = int(round(dy * resolution))

    return dynamics

# @jit(cache=True)
def kernel_loop(kernel, xs, ys, phis, n_modes, dynamics):
    previous_kernel = np.copy(kernel)
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(phis)):
                    if kernel[i, j, k] == 1:
                        continue 
                    kernel[i, j, k] = check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel, xs, ys)

    return kernel

@njit(cache=True)
def check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel, xs, ys):
    n_pts = 5
    for l in range(n_modes):
        for t in range(n_pts):
            di, dj, new_k = dynamics[k, l, t, :]
            new_i = min(max(0, i + di), len(xs)-1)  
            new_j = min(max(0, j + dj), len(ys)-1)

            if previous_kernel[new_i, new_j, new_k]:
                # if you hit a constraint, break
                break

            if not previous_kernel[new_i, new_j, new_k] and t == n_pts - 1:
                return False

    return True


def run_original():
    viab = ViabilityKernel()
    # viab.load_kernel()
    # viab.view_kernel(0)
    viab.calculate_kernel(20)
    # viab.view_kernel(-0.2)
    viab.view_kernel(0)
    # viab.view_kernel(0.2)
    # viab.view_all_modes(0)



if __name__ == "__main__":
    # t = timeit.timeit(run_original, number=1)
    # print(f"Time taken: {t}")
    run_original()


