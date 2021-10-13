import numpy as np
from numba import njit, jit
from matplotlib import pyplot as plt, rc_context
import timeit

from numpy.core.defchararray import mod



class ViabilityKernel:
    def __init__(self, width=1, length=2):
        self.resolution = 20
        self.t_step = 0.08
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

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step)
        self.set_mode_window_size()
        # self.build_dynamics_table()
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
        self.kernel[x_start:x_start+obs_offset, y_start:y_start + obs_offset, :, :] = 1

    def set_mode_window_size(self):
        sv = 3.2 # rad/s 
        d_delta = sv * self.t_step
        self.mode_window_size = int((d_delta / (0.8 / (self.n_modes-1))))
        print(f"Mode Window size: {self.mode_window_size}")

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.qs, self.mode_window_size, self.n_modes, self.dynamics, self.x_offset, self.y_offset, self.phi_range, self.resolution, self.half_block)

        np.save("SupervisorySafetySystem/Discrete/RelativeObsKernal.npy", self.kernel)
        print(f"Saved kernel to file")

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/RelativeObsKernal.npy")

    def view_kernel(self, phi):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        mode = 4
        img = self.kernel[:, :, phi_ind, mode].T + self.constraints[:, :, phi_ind, mode].T * 2
        plt.imshow(img, origin='lower')

        self.plot_next_state(phi)

        plt.show()

    def view_all_modes(self, phi):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        for m in range(self.n_modes):
            plt.figure()
            plt.title(f"Kernel phi: {phi} Mode: {m}")
            img = self.kernel[:, :, phi_ind, m].T + self.constraints[:, :, phi_ind, m].T *2
            plt.imshow(img, origin='lower')
        plt.show()

    def plot_next_state(self, o_phi):
        plt.figure(2)
        plt.title(f"Next state phi: {o_phi} ")
        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(o_phi)*arrow_len, np.cos(o_phi)*arrow_len, color='r', width=0.001)
        for i in range(self.n_modes):
            x, y = 0, 0 
            new_x, new_y, new_phi = self.safe_update(x, y, o_phi, i)

            plt.arrow(new_x, new_y, np.sin(new_phi)*arrow_len, np.cos(new_phi)*arrow_len, color='b', width=0.001)

        plt.show()

# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, t_step):
    dynamics = np.zeros((len(phis), len(qs), 3))

    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            dynamics[i, j, 2] = p + m * t_step
            dynamics[i, j, 0] = np.sin(dynamics[i, j, 2]) * velocity * t_step
            dynamics[i, j, 1] = np.cos(dynamics[i, j, 2]) * velocity * t_step

    return dynamics

# @jit(cache=True)
def kernel_loop(kernel, xs, ys, phis, qs, mode_window, n_modes, dynamics, x_offset, y_offset, phi_range, resolution, half_block):
    previous_kernel = np.copy(kernel)
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(phis)):
                for m in range(len(qs)):
                    if kernel[i, j, k, m] == 1:
                        continue 
                    kernel[i, j, k, m] = check_kernel_state(xs[i], ys[j], k, m, mode_window, n_modes, dynamics, previous_kernel, xs, ys, phis, x_offset, y_offset, phi_range, resolution, half_block)

    return kernel

# @njit(cache=True)
def check_kernel_state(x, y, k, m, mode_window, n_modes, dynamics, previous_kernel, xs, ys, phis, x_offset, y_offset, phi_range, resolution, half_block):
    min_m = max(0, m-mode_window)
    max_m = min(n_modes, m+mode_window+1)
    for l in range(min_m, max_m):
        dx, dy, new_phi = dynamics[k, m, :]
        new_x, new_y = x + dx, y + dy
        if new_phi <= -np.pi/2 or new_phi >= np.pi/2:
            continue

        kernal_inds = convert_state_to_kernel(new_x, new_y, new_phi, xs, ys, phis, x_offset, y_offset, phi_range, resolution, half_block)
        if not previous_kernel[kernal_inds[0], kernal_inds[1], kernal_inds[2], l] == 1:
            return False 
    return True

#TODO: remove this with better dynamics table usage.
def convert_state_to_kernel(x, y, phi, xs, ys, phis, x_offset, y_offset, phi_range, resolution, half_block):
    # x_ind = int(round((x+self.half_block-self.x_offset)*self.resolution))
    x_ind = int(round((x-x_offset)*resolution))
    x_ind = max(0, min(x_ind, len(xs)-1))
    y_ind = int((y+half_block-y_offset)*resolution)
    y_ind = max(0, min(y_ind, len(ys)-1))
    phi_ind = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))

    return (x_ind, y_ind, phi_ind)
 


def run_original():
    viab = ViabilityKernel()
    # viab.load_kernel()
    # viab.view_kernel(0)
    viab.calculate_kernel(20)
    # viab.view_kernel(-0.2)
    # viab.view_kernel(0)
    # viab.view_kernel(0.2)
    # viab.view_all_modes(0)



if __name__ == "__main__":
    t = timeit.timeit(run_original, number=1)
    print(f"Time taken: {t}")


