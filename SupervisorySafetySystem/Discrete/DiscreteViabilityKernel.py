import numpy as np
from numba import njit
from matplotlib import pyplot as plt




class ViabilityKernel:
    def __init__(self, width=1, length=5):
        self.resolution = 10 # pts per meter
        self.width = width
        self.length = length
        self.n_x = self.resolution * self.width
        self.n_y = self.resolution * self.length
        self.xs = np.linspace(0, self.width, self.resolution*self.width)
        self.ys = np.linspace(0, self.length, self.resolution*self.length)
        
        self.n_phi = 50
        self.phis = np.linspace(-np.pi, np.pi, self.n_phi)

        self.qs = None
        self.n_modes = None
        self.build_qs()

        #TODO: add modes as states later. For the moment, assume no dynamic window. All velocities are instantly reachable.
        # note: todo that I will have to have two dimensions for n_modes. One for the state and one for each possible option.

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.set_track_constraints()

    def build_qs(self):
        v_max = 5  
        d_max = 0.4  
        L = 0.33
        n_vs = 5
        v_pts = np.linspace(1, v_max, n_vs)
        d_resolution = 0.1
        n_ds = [9, 9, 7, 5, 3] # number of steering points in triangle
        #TODO: automate this so that it can auto adjust
        n_modes = int(np.sum(n_ds))
        self.n_modes = n_modes
        temp_qs = np.zeros((n_modes, 2))
        idx = 0
        for i in range(len(v_pts)): #step through vs
            for j in range(n_ds[i]): # step through ds 
                temp_qs[idx, 0] = (j - (n_ds[i] -1)/2)  * d_resolution
                temp_qs[idx, 1] = v_pts[i]
                idx += 1

        self.qs = np.zeros((n_modes, 2))
        for idx in range(n_modes):
            self.qs[idx, 1] = temp_qs[idx, 1]
            self.qs[idx, 0] = temp_qs[idx, 1] / L * np.tan(temp_qs[idx,0]) 

        # plt.figure()
        # for pt in self.qs:
        # # for pt in temp_qs:
        #     plt.plot(pt[0], pt[1], 'ro')
        # plt.show()

        return np.copy(self.qs)

    def set_track_constraints(self):
        # left and right wall
        self.kernel[0, :, :, :] = 1
        self.kernel[-1, :, :, :] = 1
        self.kernel[:, 32:35, :, :] = 1


    def safe_update(self, x, y, phi, q_input, t_step=1):
        new_x = x + np.sin(phi) * self.qs[q_input, 1] * t_step
        new_y = y + np.cos(phi) * self.qs[q_input, 1] * t_step
        new_phi = phi + self.qs[q_input, 0] * t_step

        return new_x, new_y, new_phi

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            for i in range(self.n_x):
                # print(f"Running XXXs: {i}")
                for j in range(self.n_y):
                    # print(f"Running YYYs: {j}")
                    for k in range(self.n_phi):
                        for l in range(self.n_modes):
                            self.kernel[i, j, k, l] += self.check_state(i, j, k, l)

        np.save("SupervisorySafetySystem/Discrete/ViabilityKernal.npy", self.kernel)

    def check_state(self, i, j, k, l):
        x = self.xs[i]
        y = self.ys[j]
        phi = self.phis[k]
        # q_input = self.qs[l]
        new_x, new_y, new_phi = self.safe_update(x, y, phi, l)

        kernal_inds = self.convert_state_to_kernel(new_x, new_y, new_phi)
        kernel = self.kernel[kernal_inds[0], kernal_inds[1], kernal_inds[2], l]

        return kernel

    def convert_state_to_kernel(self, x, y, phi):
        #TODO: very inefficient, replace with count_nonzero
        x_ind = np.argmin(np.abs(self.xs - x))
        y_ind = np.argmin(np.abs(self.ys - y))
        phi_ind = np.argmin(np.abs(self.phis - phi))

        return (x_ind, y_ind, phi_ind)

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/ViabilityKernal.npy")

    def view_kernel(self, phi, mode):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure()
        plt.imshow(self.kernel[:, :, phi_ind, mode].T, origin='lower')
        plt.show()


if __name__ == "__main__":
    viab = ViabilityKernel()
    # viab.load_kernel()
    # viab.view_kernel(0, 32)
    viab.calculate_kernel(1)
    # viab.view_kernel(0, 32)
    # viab.calculate_kernel(1)
    # viab.view_kernel(0, 32)
    # viab.calculate_kernel(1)
    # viab.view_kernel(0, 32)

    viab.view_kernel(0, 0)
    viab.view_kernel(0, 10)
    viab.view_kernel(0, 20)
    viab.view_kernel(0, 32)
