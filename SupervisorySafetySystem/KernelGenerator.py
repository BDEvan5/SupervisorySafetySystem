import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SupervisorySafetySystem.Simulator.Dynamics import update_std_state, update_complex_state, update_complex_state_const
from SupervisorySafetySystem.KernelTests.GeneralTestTrain import load_conf
from SupervisorySafetySystem.Modes import Modes

class KernelGenerator:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.sim_conf = sim_conf
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.phi_range = sim_conf.phi_range
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.m = Modes(sim_conf)
        self.n_modes = self.m.n_modes

        self.o_map = np.copy(self.track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        
        self.kernel[:, :, :, :] = self.track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))

        if sim_conf.kernel_mode == "viab":
            self.dynamics = build_viability_dynamics(self.phis, self.m, self.t_step, self.sim_conf)
        elif sim_conf.kernel_mode == 'disc':
            self.dynamics = build_disc_dynamics(self.phis, self.m, self.t_step, self.sim_conf)
        else:
            raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")

    def save_kernel(self, name):
        np.save(f"{self.sim_conf.kernel_path}{name}.npy", self.kernel)
        print(f"Saved kernel to file: {name} -> {self.kernel.shape}")


        self.view_speed_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelSpeed_{name}_{self.sim_conf.kernel_mode}.png")

        self.view_angle_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelAngle_{name}_{self.sim_conf.kernel_mode}.png")



    def get_filled_kernel(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")
        return filled/total

    def view_angle_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)

        mode_ind = 9

        self.axs[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
     
    def view_speed_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)
        # phi_ind = 0
        # quarter_phi = int(len(self.phis)/4)
        # phi_ind = 

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, 2].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel speed: {2}")
        # axs[0, 0].clear()
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, 6].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel speed: {3}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, 8].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel speed: {4}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, 9].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel speed: {5}")

        # plt.title(f"Building Kernel")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
    
    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)


        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.sim_conf.kernel_path}Kernel_build_{self.sim_conf.kernel_mode}.svg")

        if show:
            plt.show()

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics)

            # self.view_kernel(0, False)
            self.view_speed_build(False)

        return self.get_filled_kernel()



# @njit(cache=True)
def build_viability_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range

    ns = 2

    dynamics = np.zeros((len(phis), len(m), len(m), ns, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time/2)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 0, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 0, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 0, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 0, 3] = int(new_q)                  
                

                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 1, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 1, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 1, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 1, 3] = int(new_q)                  
                

    return dynamics


# @njit(cache=True)
def build_disc_dynamics(phis, m, time, conf):
    resolution = conf.n_dx
    phi_range = conf.phi_range
    block_size = 1 / (resolution)
    h = conf.discrim_block * block_size *0.5
    phi_size = phi_range / (conf.n_phi -1)
    ph = conf.discrim_phi * phi_size

    ns = 1
    dynamics = np.zeros((len(phis), len(m), len(m), 9, 4), dtype=np.int)
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(m.qs): # searches through actions
                new_state = update_complex_state(state, action, time)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi < -np.pi:
                    phi = phi + 2*np.pi

                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                temp_dynamics = generate_temp_dynamics(dx, dy, h, resolution)
                
                dynamics[i, j, k, 0:8, 0:2] = np.copy(temp_dynamics)
                dynamics[i, j, k, 0:8, 3] = int(new_q) # no q discretisation error

                new_state = update_complex_state(state, action, time/2)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = m.get_mode_id(vel, steer)

                while phi > np.pi:
                    phi = phi - 2*np.pi
                while phi < -np.pi:
                    phi = phi + 2*np.pi
                new_k = int(round((phi + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, k, 8, 2] = min(max(0, new_k), len(phis)-1)
                
                dynamics[i, j, k, 8, 0] = int(round(dx * resolution))                  
                dynamics[i, j, k, 8, 1] = int(round(dy * resolution))                  
                dynamics[i, j, k, 8, 3] = int(new_q)   

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


@njit(cache=True)
def viability_loop(kernel, dynamics):
    previous_kernel = np.copy(kernel)
    l_xs, l_ys, l_phis, l_qs = kernel.shape
    for i in range(l_xs):
        for j in range(l_ys):
            for k in range(l_phis):
                for q in range(l_qs):
                    if kernel[i, j, k, q] == 1:
                        continue 
                    kernel[i, j, k, q] = check_viable_state(i, j, k, q, dynamics, previous_kernel)

    return kernel


@njit(cache=True)
def check_viable_state(i, j, k, q, dynamics, previous_kernel):
    l_xs, l_ys, l_phis, n_modes = previous_kernel.shape
    for l in range(n_modes):
        safe = True
        for n in range(dynamics.shape[3]): # cycle through 8 block states
            di, dj, new_k, new_q = dynamics[k, q, l, n, :]
            new_i = min(max(0, i + di), l_xs-1)  
            new_j = min(max(0, j + dj), l_ys-1)

            if previous_kernel[new_i, new_j, new_k, new_q]:
                # if you hit a constraint, break
                safe = False # breached a limit.
                break # try again and look for a new action

        if safe: # there exists a valid action
            return False # it is safe

    return True # it isn't safe because I haven't found a valid action yet...


"""
    External functions
"""

def prepare_track_img(sim_conf):
    file_name = 'maps/' + sim_conf.map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)
        yaml_file = dict(documents.items())
    img_resolution = yaml_file['resolution']
    map_img_path = 'maps/' + yaml_file['image']

    resize = int(sim_conf.n_dx * img_resolution)

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


@njit(cache=True)
def shrink_img(img, n_shrinkpx):
    o_img = np.copy(img)

    search = np.array([[0, 1], [1, 0], [0, -1], 
                [-1, 0], [1, 1], [1, -1], 
                [-1, 1], [-1, -1]])
    for i in range(n_shrinkpx):
        t_img = np.copy(img)
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if img[j, k] == 1:
                    continue
                for l in range(len(search)):
                    di, dj = search[l, :]
                    new_i = min(max(0, j + di), img.shape[0]-1)
                    new_j = min(max(0, k + dj), img.shape[1]-1)
                    if t_img[new_i, new_j] == 1:
                        img[j, k] = 1.
                        break

    print(f"Finished Shrinking")
    return o_img, img #




def build_track_kernel(conf):
  
    img = prepare_track_img(conf) 
    # img, img2 = shrink_img(img, 5)
    kernel = KernelGenerator(img, conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_viab_{conf.map_name}")
    # kernel.view_build(True)




if __name__ == "__main__":
    conf = load_conf("std_test_kernel")
    build_track_kernel(conf)


