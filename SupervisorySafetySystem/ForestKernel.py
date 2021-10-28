import numpy as np
from numba import njit
from matplotlib import pyplot as plt



class ForestKernelGenerator:
    def __init__(self, track_img, sim_conf):
        self.velocity = 2 #TODO: make this a config param
        self.track_img = track_img
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.time_step
        self.n_phi = sim_conf.n_phi
        self.phi_range = sim_conf.phi_range
        self.half_block = 1 / (2*self.n_dx)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = sim_conf.n_modes
        self.sim_conf = sim_conf

        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = track_img.shape[0]
        self.n_y = track_img.shape[1]
        self.xs = np.linspace(0, sim_conf.forest_width, self.n_x)
        self.ys = np.linspace(0, sim_conf.forest_length, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.sim_conf)

        self.kernel[:, :, :] = track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

    # config functions
    def build_qs(self):
        ds = np.linspace(-self.max_steer, self.max_steer, self.n_modes)
        self.qs = self.velocity / self.L * np.tan(ds)

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

    def save_kernel(self, name="std_kernel"):
        np.save(f"SupervisorySafetySystem/Kernels/{name}.npy", self.kernel)
        print(f"Saved kernel to file")

    def view_kernel(self, phi, show=True):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
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


def update_dynamics(phi, th_dot, velocity, time_step):
    new_phi = phi + th_dot * time_step
    dx = np.sin(phi) * velocity * time_step
    dy = np.cos(phi) * velocity * time_step

    return dx, dy, new_phi

# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, conf):
    resolution = conf.n_dx
    n_pts = conf.dynamics_pts
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size 
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    dynamics = np.zeros((len(phis), len(qs), n_pts, 8, 3), dtype=np.int)
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            for t in range(n_pts): 
                t_step = time * (t+1)  / n_pts
                dx, dy, phi = update_dynamics(p, m, velocity, t_step)

                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                temp_dynamics = generate_temp_dynamics(dx, dy, h, resolution)
                
                dynamics[i, j, t, :, 0:2] = np.copy(temp_dynamics)

                pass

    return dynamics

@njit(cache=True)
def generate_temp_dynamics(dx, dy, h, resolution):
    temp_dynamics = np.zeros((8, 2))

    for i in range(2):
        temp_dynamics[0 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[0 + i*4, 1] = int(round((dy -h) * resolution))
        temp_dynamics[1 + i*4, 0] = int(round((dx -h) * resolution))
        temp_dynamics[1 + i*4, 1] = int(round((dy +h) * resolution))
        temp_dynamics[2 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[2 + i*4, 1] = int(round((dy +h )* resolution))
        temp_dynamics[3 + i*4, 0] = int(round((dx +h) * resolution))
        temp_dynamics[3 + i*4, 1] = int(round((dy -h) * resolution))

    return temp_dynamics

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
    n_pts = dynamics.shape[2]
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
    img_size = int(conf.obs_img_size * conf.n_dx)
    obs_size = int(conf.obs_size * conf.n_dx)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = ForestKernelGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def construct_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) * conf.n_dx , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = ForestKernelGenerator(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")



