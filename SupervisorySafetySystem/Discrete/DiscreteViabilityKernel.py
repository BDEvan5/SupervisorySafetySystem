import numpy as np
from numba import njit
from matplotlib import pyplot as plt




class ViabilityKernel:
    def __init__(self, width=1, length=5):
        self.resolution = 20 # pts per meter
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
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.set_track_constraints()

    def build_qs(self):
        L = 0.33
        
        self.n_modes = 9
        ds = np.linspace(-0.4, 0.4, self.n_modes)
        self.qs = self.velocity / L * np.tan(ds)

    def set_track_constraints(self):
        # left and right wall
        self.constraints[0, :, :] = 1
        self.constraints[-1, :, :] = 1
        n = int(self.resolution / 3)

        # self.constraints[n:-n, 3*self.resolution:self.resolution*3+2*n, :] = 1

        self.constraints[0:n, 2*self.resolution:2*self.resolution+2*n, :] = 1
        self.constraints[-n::, 4*self.resolution:self.resolution*4+2*n, :] = 1


        self.kernel = np.copy(self.constraints)

    def safe_update(self, x, y, phi, q_input, t_step=0.08):
        new_phi = phi + self.qs[q_input] * t_step
        new_x = x + np.sin(new_phi) * self.velocity * t_step
        new_y = y + np.cos(new_phi) * self.velocity * t_step

        return new_x, new_y, new_phi

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            for i in range(self.n_x):
                for j in range(self.n_y):
                    for k in range(self.n_phi):
                        if self.kernel[i, j, k] == 1:
                            continue 
                        self.kernel[i, j, k] = self.update_state(i, j, k)

        np.save("SupervisorySafetySystem/Discrete/ViabilityKernal.npy", self.kernel)
        print(f"Saved kernel to file")

    def update_state(self, i, j, k):
        # TODO: add in checking only certain modes here
        for l in range(self.n_modes):
            if not self.check_state(i, j, k, l):
                return 0
        return 1

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
        if y < 0:
            return True
        if phi < -self.phi_range/2 or phi >= self.phi_range/2:
            return True
        return False

    def convert_state_to_kernel(self, x, y, phi):
        x_ind = int(x*self.resolution)
        y_ind = int(y*self.resolution)
        y_ind = min(y_ind, len(self.ys)-1)
        phi_ind = int((phi + self.phi_range/2) / self.phi_range * self.n_phi)

        return (x_ind, y_ind, phi_ind)

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/ViabilityKernal.npy")

    def view_kernel(self, phi):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        plt.imshow(self.kernel[:, :, phi_ind].T, origin='lower')

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
    viab = ViabilityKernel()
    # viab.load_kernel()
    # viab.view_kernel(0, 8)
    viab.calculate_kernel(20)
    viab.view_kernel(-0.2)
    viab.view_kernel(0)
    viab.view_kernel(0.2)


