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
        
        self.n_phi = 20
        self.phi_range = np.pi
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)

        self.velocity = 2
        self.qs = None
        self.n_modes = None
        self.build_qs()

        #TODO: add modes as states later. For the moment, assume no dynamic window. All velocities are instantly reachable.
        # note: todo that I will have to have two dimensions for n_modes. One for the state and one for each possible option.

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.constraints = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.set_track_constraints()

    def build_qs(self):
        L = 0.33
        
        ds = np.linspace(-0.4, 0.4, 9)
        self.qs = self.velocity / L * np.tan(ds)
        self.n_modes = 9

    def set_track_constraints(self):
        # left and right wall
        self.constraints[0, :, :] = 1
        self.constraints[-1, :, :] = 1
        self.constraints[1:3, 32:35, :] = 1

        self.kernel = np.copy(self.constraints)

    def safe_update(self, x, y, phi, q_input, t_step=0.1):
        new_x = x + np.sin(phi) * self.velocity * t_step
        new_y = y + np.cos(phi) * self.velocity * t_step
        new_phi = phi + self.qs[q_input] * t_step

        return new_x, new_y, new_phi

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            for i in range(self.n_x):
                # print(f"Running XXXs: {i}")
                for j in range(self.n_y):
                    # print(f"Running YYYs: {j}")
                    for k in range(self.n_phi):
                        if self.kernel[i, j, k] == 1:
                            continue 
                        self.kernel[i, j, k] = 1 # set it to unviable.
                        for l in range(self.n_modes):
                            if not self.check_state(i, j, k, l):
                                self.kernel[i, j, k] = 0
                                break # can stop checking now.


        np.save("SupervisorySafetySystem/Discrete/ViabilityKernal.npy", self.kernel)
        print(f"Saved kernel to file")

    def check_state(self, i, j, k, l):
        x = self.xs[i]
        y = self.ys[j]
        phi = self.phis[k]
        new_x, new_y, new_phi = self.safe_update(x, y, phi, l)

        if self.check_limits(new_x, new_y, new_phi):
            return True 

        kernal_inds = self.convert_state_to_kernel(new_x, new_y, new_phi)
        kernel = self.kernel[kernal_inds[0], kernal_inds[1], kernal_inds[2]]

        return kernel

    def check_limits(self, x, y, phi):
        if x < 0 or x > self.width:
            return True
        if y < 0 or y > self.length:
            return True
        if phi < -self.phi_range/2 or phi > self.phi_range/2:
            return True
        return False

    def convert_state_to_kernel(self, x, y, phi):
        x_ind = int(x*(self.resolution-1))
        y_ind = int(y*(self.resolution-1))
        phi_ind = int((phi + self.phi_range/2) / self.phi_range * (self.n_phi-1))

        return (x_ind, y_ind, phi_ind)

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/ViabilityKernal.npy")

    def view_kernel(self, phi):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        plt.imshow(self.kernel[:, :, phi_ind].T, origin='lower', extent=[0, 10, 0, self.length*self.resolution])

        # plt.imshow(self.constraints[:, :, 0].T, origin='lower', extent=[-20, -10, 0, self.length*self.resolution])

        self.plot_next_state(phi)

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


if __name__ == "__main__":
    # viab = ViabilityKernel()
    # viab.load_kernel()
    # viab.view_kernel(0, 8)
    # viab.calculate_kernel(1)
    # viab.view_kernel(0)
    # viab.view_kernel(-np.pi/3)
    # viab.view_kernel(np.pi/3)
    # viab.view_kernel(0)
    # viab.calculate_kernel(1)
    # viab.view_kernel(0, 32)
    # viab.calculate_kernel(1)
    # viab.view_kernel(0, 32)

    # viab.view_kernel(-np.pi/4)
    # viab.view_kernel(0)
    # viab.view_kernel(0, 4)
    # viab.view_kernel(0, 6)

    # viab.plot_next_state(np.pi/6)


