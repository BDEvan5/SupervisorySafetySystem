import numpy as np
from numba import njit
from matplotlib import pyplot as plt




class ViabilityKernel:
    def __init__(self, width=1, length=2):
        self.resolution = int(20) # pts per meter
        self.t_step = 0.08
        self.velocity = 2
        self.n_phi = 20

        self.n_x = self.resolution * width
        self.n_y = self.resolution * length
        self.x_offset = -0.25 
        self.y_offset = -1.5
        self.xs = np.linspace(self.x_offset, self.x_offset + width, self.n_x)
        self.ys = np.linspace(self.y_offset, self.y_offset + length, self.n_y)
        
        self.phi_range = np.pi
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)

        self.qs = None
        self.n_modes = None
        self.build_qs()
        self.set_mode_window_size()

        #TODO: add modes as states later. For the moment, assume no dynamic window. All velocities are instantly reachable.
        # note: todo that I will have to have two dimensions for n_modes. One for the state and one for each possible option.

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.constraints = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.set_track_constraints()

    def build_qs(self):
        L = 0.33
        
        self.n_modes = 9
        ds = np.linspace(-0.4, 0.4, self.n_modes)
        self.qs = self.velocity / L * np.tan(ds)

    def set_track_constraints(self):
        obs_size = 0.5 
        obs_offset = int(obs_size*self.resolution)
        x_start = int(-self.x_offset*self.resolution)
        y_start = int(-self.y_offset*self.resolution)
        self.constraints[x_start:x_start+obs_offset, y_start:y_start + obs_offset, :, :] = 1

        self.kernel = np.copy(self.constraints)

    def safe_update(self, x, y, phi, q_input):
        new_phi = phi + self.qs[q_input] * self.t_step
        new_x = x + np.sin(new_phi) * self.velocity * self.t_step
        new_y = y + np.cos(new_phi) * self.velocity * self.t_step

        return new_x, new_y, new_phi

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
            for i in range(self.n_x):
                for j in range(self.n_y):
                    for k in range(self.n_phi):
                        for m in range(self.n_modes):
                            if self.kernel[i, j, k, m] == 1:
                                continue 
                            self.kernel[i, j, k, m] = self.update_state(i, j, k, m)

        np.save("SupervisorySafetySystem/Discrete/RelativeObsKernal.npy", self.kernel)
        print(f"Saved kernel to file")

    def update_state(self, i, j, k, m):
        # TODO: add in checking only certain modes here
        min_m = max(0, m-self.mode_window_size)
        max_m = min(self.n_modes, m+self.mode_window_size)
        for l in range(min_m, max_m):
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
        kernel = self.kernel[kernal_inds[0], kernal_inds[1], kernal_inds[2], l]

        return kernel

    def check_limits(self, x, y, phi):
        if x < self.x_offset or x > self.xs[-1]:
            return True
        if y < self.y_offset:
            return True
        if phi < -self.phi_range/2 or phi >= self.phi_range/2:
            return True
        return False

    def convert_state_to_kernel(self, x, y, phi):
        x_ind = int((x-self.x_offset)*self.resolution)
        y_ind = int((y-self.y_offset)*self.resolution)
        y_ind = min(y_ind, len(self.ys)-1)
        phi_ind = int((phi + self.phi_range/2) / self.phi_range * self.n_phi)

        return (x_ind, y_ind, phi_ind)

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/RelativeObsKernal.npy")

    def view_kernel(self, phi):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        plt.imshow(self.kernel[:, :, phi_ind, int((self.n_modes-1)/2)].T, origin='lower')

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
    viab.view_kernel(0)
    viab.calculate_kernel(20)
    viab.view_kernel(-0.2)
    viab.view_kernel(0)
    viab.view_kernel(0.2)


