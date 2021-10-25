import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image


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


class TrackKernel:
    def __init__(self, sim_conf):
        self.velocity = 2 #TODO: make this a config param
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

        self.track_img = self.prepare_img()

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, 16.25, self.n_x)
        self.ys = np.linspace(0, 6, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.sim_conf)

        self.kernel[:, :, :] = self.track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))

    def prepare_img(self):
        file_name = 'maps/' + self.sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            img_resolution = yaml_file['resolution']
            map_img_path = 'maps/' + yaml_file['image']
        except Exception as e:
            print(f"Problem loading, check key: {e}")
            raise FileNotFoundError("Problem loading map yaml file")

        map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        map_img = map_img.astype(np.float64)
        if len(map_img.shape) == 3:
            map_img = map_img[:, :, 0]
        map_img[map_img <= 128.] = 1.
        map_img[map_img > 128.] = 0.

        # porto crop vals.
        # crop_x = [50, 375]
        # crop_y = [200, 320]
        map_img = map_img.T
        # map_img = map_img[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]

        self.map_height = map_img.shape[0]
        self.map_width = map_img.shape[1]

        img = Image.fromarray(map_img)
        scale = 4
        img = img.resize((self.map_width*scale, self.map_height*scale))
        img = np.array(img)
        map_img = img.astype(np.float64)
        map_img[map_img != 0.] = 1.

        # plt.figure(1)
        # plt.imshow(map_img.T, origin='lower')
        # plt.show()

        self.o_map = map_img
        # start with keeping resolution the same.
        return map_img

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
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

            self.view_kernel(0, False, 1)
            self.view_kernel(-np.pi/2, False, 2)
            self.view_kernel(np.pi/2, False, 3)

    def save_kernel(self, name="track_std_kernel"):
        np.save(f"SupervisorySafetySystem/Kernels/{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name}")

    def view_kernel(self, phi, show=True, fig_n=1):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(fig_n)
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

# @jit(cache=True)
l_xs = 1300
l_ys = 480
l_phis = 41

@njit(cache=True)
def kernel_loop(kernel, xs, ys, phis, n_modes, dynamics):
    previous_kernel = np.copy(kernel)
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
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
            di, dj, new_k = dynamics[k, l, t, :]
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



from SupervisorySafetySystem.KernelTests.GeneralTestTrain import load_conf
def build_track_kernel():
    conf = load_conf("track_kernel")

    kernel = TrackKernel(conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"TrackKernel_{conf.track_kernel_path}")


if __name__ == "__main__":
    build_track_kernel()


