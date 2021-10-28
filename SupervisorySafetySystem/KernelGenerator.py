import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image

from SupervisorySafetySystem.KernelTests.GeneralTestTrain import load_conf


class ViabilityGenerator:
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

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) #TODO: magic number what!!!!
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.sim_conf)
        self.o_map = np.copy(self.track_img)    
        self.kernel[:, :, :] = self.track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))


    # config functions
    def build_qs(self):
        ds = np.linspace(-self.max_steer, self.max_steer, self.n_modes)
        self.qs = self.velocity / self.L * np.tan(ds)

    def calculate_kernel(self, n_loops=20):
        # self.view_kernel(0)
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.n_modes, self.dynamics)

            self.view_kernel(0, False, 1)
            self.view_kernel(-np.pi/2, False, 2)
            self.view_kernel(np.pi/2, False, 3)

    def save_kernel(self, name):
        np.save(f"SupervisorySafetySystem/Kernels/{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name}")

    def view_kernel(self, phi, show=True, fig_n=1):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(fig_n)
        plt.clf()
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        img = self.kernel[:, :, phi_ind].T + self.o_map.T
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

    dynamics = np.zeros((len(phis), len(qs), n_pts, 3), dtype=np.int)
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            for t in range(n_pts): 
                t_step = time * (t+1)  / n_pts
                dx, dy, phi = update_dynamics(p, m, velocity, t_step)

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1))) # TODO: check that i gets around the circle.
                dynamics[i, j, t, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, t, 0] = int(round(dx * resolution))                  
                dynamics[i, j, t, 1] = int(round(dy * resolution))                  
                

    return dynamics


@njit(cache=True)
def kernel_loop(kernel, n_modes, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                    if kernel[i, j, k] == 1:
                        continue 
                    kernel[i, j, k] = check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel)

    return kernel

@njit(cache=True)
def check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel):
    n_pts = dynamics.shape[2]
    l_xs, l_ys, l_phis = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        # check all concatanation points and offsets and if none are occupied, then it is safe.
        for t in range(n_pts):
            di, dj, new_k = dynamics[k, l, t, :]
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

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


def prepare_track_img(sim_conf, resize=5):
    file_name = 'maps/' + sim_conf.map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    img_resolution = yaml_file['resolution']
    map_img_path = 'maps/' + yaml_file['image']

    map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    map_img = map_img.astype(np.float64)
    if len(map_img.shape) == 3:
        map_img = map_img[:, :, 0]
    map_img[map_img <= 128.] = 1.
    map_img[map_img > 128.] = 0.

    img = Image.fromarray(map_img.T)
    img = img.resize((map_img.shape[0]*resize, map_img.shape[1]*resize))
    img = np.array(img)
    map_img2 = img.astype(np.float64)
    map_img2[map_img2 != 0.] = 1.

    return map_img2



def build_track_kernel():
    conf = load_conf("track_kernel")

    img = prepare_track_img(conf, 4) #NB change this param to set the difference between the map resolution and the kernel resolution. 0.05 -> 80 ndx is good for now. 
    # plt.figure(1)
    # plt.imshow(img)
    # plt.pause(0.0001)
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(30)
    kernel.save_kernel(f"TrackKernel_{conf.track_kernel_path}_{conf.map_name}")


if __name__ == "__main__":
    build_track_kernel()


