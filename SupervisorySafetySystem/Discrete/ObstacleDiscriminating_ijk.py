import numpy as np
from numba import njit, jit
from matplotlib import pyplot as plt, rc_context
import timeit

from numpy.core.defchararray import mod



class DiscriminatingKernel:
    def __init__(self, width=1, length=2):
        self.resolution = 200
        self.t_step = 0.2
        self.velocity = 2
        self.n_phi = 61
        self.phi_range = np.pi
        self.half_block = 1 / (2*self.resolution)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = 5

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
        max_steer = 0.35
        ds = np.linspace(-max_steer, max_steer, self.n_modes)
        self.qs = self.velocity / 0.33 * np.tan(ds)

    def set_track_constraints(self):
        obs_size = 0.5 
        obs_buff = int(0.05*self.resolution)
        obs_offset = int(obs_size*self.resolution)
        x_start = int(-self.x_offset*self.resolution)
        y_start = int(-self.y_offset*self.resolution)
        self.kernel[x_start-obs_buff:x_start+obs_offset+obs_buff, y_start-obs_buff:y_start + obs_offset, :] = 1



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
            phi_n = 30
            img = self.kernel[:, :, phi_n].T - self.previous_kernel[:, :, phi_n].T
            plt.imshow(img, origin='lower')
            plt.pause(0.0001)

            self.view_kernel(0, False)
        self.save_kernel()

    def save_kernel(self):
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
            di, dj, new_k = self.dynamics[phi_ind, m, 0,-1]


            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)
        if show:
            plt.show()

    def linear_interp(self):
        for p in range(self.n_phi): 
            pts = []
            prev_j = 1000
            prev_i = 0
            center = False
            half_point = int(self.n_x/2)
            for i in range(half_point):
                if not np.any(self.kernel[i, :, p]):
                    continue
                for j in range(self.n_y):
                    if self.kernel[i, j, p]:
                        if abs(j - prev_j) < 2:
                            break
                        if not center:
                            pts.append((i, j))
                            prev_j = j
                            break

            for i in range(half_point, self.n_x):
                if not np.any(self.kernel[i, :, p]):
                    continue
                for j in range(self.n_y):
                    if self.kernel[i, j, 10]:
                        if j == self.n_y - 1:
                            break
                        print(f"{i}, {j}")
                        prev_i = i
                        if abs(j - prev_j) > 2:
                            pts.append((i-1, prev_j))
                            prev_j = j
                            break
                        break
            pts.append((prev_i, prev_j))

            pts = np.array(pts)
            print(pts)
            for i in range(len(pts)-1):
                dy = pts[i+1, 1] - pts[i, 1]
                if dy == 0:
                    continue
                dx = pts[i+1, 0] - pts[i, 0]
                m = int(dy/dx)
                o_x, o_y = pts[i]
                for x in range(dx):
                    if m < 0:
                        x1 = x+1
                        for y in range(-m*x1):
                            self.kernel[o_x + x1, o_y-y-1, p] = 1
                    elif m > 0:
                        y_val = m*(dx-x)
                        for y in range(y_val):
                            i_x = o_x + x 
                            i_y = o_y - y +2+ dx*m # add intercept
                            self.kernel[i_x, i_y, p] = 1

        # self.view_kernel(0, False)
        # plt.figure(1)
        # plt.plot(pts[:, 0], pts[:, 1], 'x', markersize=20)
        # plt.pause(0.0001)

        # plt.figure(3)
        # plt.imshow(new_kernel[:, :].T, origin='lower')
        # plt.plot(pts[:, 0], pts[:, 1], 'x', markersize=20)
        # plt.show()
            
    def run_random_test(self, n):
        np.random.seed(0)

        rands = np.random.random((n, 3))
        states = np.zeros_like(rands)
        states[:, 0] = rands[:, 0] * 1
        states[:, 1] = rands[:, 1] * 2
        states[:, 2] = (rands[:, 2] * np.pi) - np.pi/2

        action_set = np.linspace(-0.4, 0.4, 6)

        for test_n in range(n):
            oi, oj, ok = self.get_indices(states[test_n])
            if self.kernel[oi, oj, ok] == 1:
                continue # if I am already in the kernel then don't do anything  
            
            option = False
            for action in action_set:
                new_state = self.get_new_state(states[test_n], action)
                i, j, k = self.get_indices(new_state)
                if self.kernel[i, j, k] != 1:
                    option = True 
                
            if not option:
                print(f"State: {states[i]} --> {oi}, {oj}, {ok}")
                self.view_kernel(states[test_n, 2], False)
                plt.plot(oi, oj, 'x', markersize=30)
                # plt.plot(i, j, 'x', markersize=20)
                plt.show()

        print(f"Tests complete")

    
    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.kernel.shape[2]-1)

        return x_ind, y_ind, theta_ind

    def get_new_state(self, state, action):
        """
        Updates the state based on the action

        Args:
            state: (x, y, theta)
            action (float): steering angle
            dt: time step
        """
        L = 0.33
        velocity = 2
        theta_update = state[2] +  ((velocity / L) * np.tan(action) * self.t_step)
        dx = np.array([velocity * np.sin(theta_update),
                    velocity*np.cos(theta_update),
                    velocity / L * np.tan(action)])

        return state + dx * self.t_step


# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, resolution):
    # add 5 sample points
    block_size = 1 / (resolution)
    h = 1 * block_size
    phi_size = np.pi / (len(phis) -1)
    ph = 0.1 * phi_size
    n_pts = 5
    dynamics = np.zeros((len(phis), len(qs), n_pts, 8, 3), dtype=np.int)
    phi_range = np.pi
    n_steps = 1
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            for t in range(n_pts):
                t_step = time * (t+1)  / n_pts
                phi = p + m * t_step * n_steps # phi must be at end
                dx = np.sin(phi) * velocity * t_step
                dy = np.cos(phi) * velocity * t_step
                
                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                dynamics[i, j, t, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                dynamics[i, j, t, 0, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 0, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, t, 1, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 1, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, t, 2, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 2, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, t, 3, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 3, 1] = int(round((dy -h) * resolution))

                dynamics[i, j, t, 4, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 4, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, t, 5, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 5, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, t, 6, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 6, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, t, 7, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 7, 1] = int(round((dy -h) * resolution))

                pass

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
        safe = True
        # check all concatanation points and offsets and if none are occupied, then it is safe.
        for t in range(n_pts):
            for n in range(dynamics.shape[3]):
                di, dj, new_k = dynamics[k, l, t, n, :]
                new_i = min(max(0, i + di), len(xs)-1)  
                new_j = min(max(0, j + dj), len(ys)-1)

                if previous_kernel[new_i, new_j, new_k]:
                    # if you hit a constraint, break
                    safe = False # breached a limit.
                    break

            # if not previous_kernel[new_i, new_j, new_k] and t == n_pts - 1:
            #     return False
        if safe:
            return False

    return True


    


def run_original():
    viab = DiscriminatingKernel()
    # viab.load_kernel()
    # viab.view_kernel(0, True)
    # viab.linear_interp()
    # viab.save_kernel()
    # viab.view_kernel(0, True)
    # viab.view_kernel(0)
    viab.calculate_kernel(20)
    # viab.view_kernel(-0.2)
    # viab.view_kernel(0.2)
    # viab.view_all_modes(0)

def test_kernel():
    kern = DiscriminatingKernel()
    # kern.load_kernel()
    kern.calculate_kernel(20)

    kern.run_random_test(100000)


if __name__ == "__main__":
    # t = timeit.timeit(run_original, number=1)
    # print(f"Time taken: {t}")
    # run_original()
    test_kernel()


