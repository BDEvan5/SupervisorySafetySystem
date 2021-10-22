import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit(cache=True)
def update_state(state, action, dt):
    """
    Updates x, y, th pos accoridng to th_d, v
    """
    L = 0.33
    theta_update = state[2] +  ((action[1] / L) * np.tan(action[0]) * dt)
    dx = np.array([action[1] * np.sin(theta_update),
                action[1]*np.cos(theta_update),
                action[1] / L * np.tan(action[0])])

    return state + dx * dt 


class DiscriminatingImgKernel:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.resolution = int(1/sim_conf.resolution)
        self.t_step = sim_conf.time_step
        self.velocity = 2 #TODO: make this a config param
        self.n_phi = 61  #TODO: add to conf file
        self.phi_range = np.pi #TODO: add to conf file
        self.half_block = 1 / (2*self.resolution)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = 5 #TODO: add to conf file

        self.n_x = track_img.shape[0]
        self.n_y = track_img.shape[1]
        self.xs = np.linspace(0, 2, self.n_x)
        self.ys = np.linspace(0, 25, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.resolution)

        self.kernel[:, :, :] = track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

    # config functions
    def build_qs(self):
        max_steer = 0.35
        ds = np.linspace(-max_steer, max_steer, self.n_modes)
        self.qs = self.velocity / 0.33 * np.tan(ds)

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

            # plt.figure(2)
            # plt.title(f"Kernel after loop: {z}")
            # phi_n = 30
            # img = self.kernel[:, :, phi_n].T - self.previous_kernel[:, :, phi_n].T
            # plt.imshow(img, origin='lower')
            # plt.pause(0.0001)

            # self.view_kernel(0, False)
        # self.save_kernel()

    def save_kernel(self, name="std_kernel"):
        np.save(f"SupervisorySafetySystem/Kernels/{name}.npy", self.kernel)
        print(f"Saved kernel to file")

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


# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, resolution):
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
                #TODO: I somehow want to extricate the dynamics. I want to be able to use an external set of dynamics here....
                t_step = time * (t+1)  / n_pts
                phi = p + m * t_step * n_steps # phi must be at end
                dx = np.sin(phi) * velocity * t_step
                dy = np.cos(phi) * velocity * t_step
                
                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                dynamics[i, j, t, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                #TODO: add some looping here...
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
        if safe:
            return False

    return True


"""
    External functions

"""

def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size / conf.resolution)
    obs_size = int(conf.obs_size / conf.resolution)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = DiscriminatingImgKernel(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) / conf.resolution , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = DiscriminatingImgKernel(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")




