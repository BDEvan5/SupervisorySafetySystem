
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml, csv
from SupervisorySafetySystem.Simulator.Dynamics import update_complex_state, update_std_state


class SafetyHistory:
    def __init__(self):
        self.planned_actions = []
        self.safe_actions = []

    def add_locations(self, planned_action, safe_action=None):
        self.planned_actions.append(planned_action)
        if safe_action is None:
            self.safe_actions.append(planned_action)
        else:
            self.safe_actions.append(safe_action)

    def plot_safe_history(self):
        plt.figure(5)
        plt.clf()
        plt.title("Safe History")
        plt.plot(self.planned_actions, color='blue')
        plt.plot(self.safe_actions, '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, path, name):
        plt.figure(5)
        plt.clf()
        plt.title(f"Safe History: {name}")
        plt.plot(self.planned_actions, color='blue')
        plt.plot(self.safe_actions, color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        plt.savefig(f"{path}/{name}_actions.png")

        data = []
        for i in range(len(self.planned_actions)):
            data.append([i, self.planned_actions[i], self.safe_actions[i]])
        full_name = path + f'/{name}_training_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)


        self.planned_actions = []
        self.safe_actions = []


class Supervisor:
    def __init__(self, planner, kernel, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        self.d_max = conf.max_steer
        self.v = 2
        self.kernel = kernel
        self.planner = planner
        self.safe_history = SafetyHistory()
        self.intervene = False

        self.time_step = conf.lookahead_time_step

        # aliases for the test functions
        try:
            self.n_beams = planner.n_beams
        except: pass
        self.plan_act = self.plan
        self.name = planner.name


    def plan(self, obs):
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        safe, next_state = check_init_action(state, init_action, self.kernel, self.time_step)
        if safe:
            self.safe_history.add_locations(init_action[0], init_action[0])
            return init_action

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel, self.time_step)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            # plt.show()
            return init_action
        
        action = modify_action(valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")
        self.safe_history.add_locations(init_action[0], action[0])


        return action

    def generate_dw(self):
        n_segments = 5
        dw = np.ones((5, 2))
        dw[:, 0] = np.linspace(-self.d_max, self.d_max, n_segments)
        dw[:, 1] *= self.v
        # dw = np.vstack((dw, dw, dw))
        # dw[0:5, 1] *= 1
        # dw[5:10, 1] *= 2
        # dw[10:, 1] *= 3
        return dw


class LearningSupervisor(Supervisor):
    def __init__(self, planner, kernel, conf):
        Supervisor.__init__(self, planner, kernel, conf)
        self.intervention_mag = 0
        self.calculate_reward = None # to be replaced by a function
        self.ep_interventions = 0
        self.intervention_list = []

    def done_entry(self, s_prime):
        s_prime['reward'] = self.calculate_reward(self.intervention_mag)
        self.planner.done_entry(s_prime)
        self.intervention_list.append(self.ep_interventions)
        self.ep_interventions = 0

    def save_intervention_list(self):
        full_name = self.planner.path + f'/{self.planner.name}_intervention_list.csv'
        data = []
        for i in range(len(self.intervention_list)):
            data.append([i, self.intervention_list[i]])
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        plt.figure(6)
        plt.clf()
        plt.plot(self.intervention_list)
        plt.savefig(f"{self.planner.path}/{self.planner.name}_interventions.png")
        plt.savefig(f"{self.planner.path}/{self.planner.name}_interventions.svg")

    def plan(self, obs):
        obs['reward'] = self.calculate_reward(self.intervention_mag)
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        safe, next_state = check_init_action(state, init_action, self.kernel, self.time_step)
        if safe:
            self.intervention_mag = 0
            self.safe_history.add_locations(init_action[0], init_action[0])
            return init_action

        self.ep_interventions += 1
        self.intervene = True

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel, self.time_step)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            # plt.show()
            self.intervention_mag = 1
            return init_action
        
        action = modify_action(valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")
        self.safe_history.add_locations(init_action[0], action[0])

        self.intervention_mag = action[0] - init_action[0]

        return action


#TODO jit all of this.

def check_init_action(state, u0, kernel, time_step=0.1):
    next_state = update_complex_state(state, u0, time_step)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

def simulate_and_classify(state, dw, kernel, time_step=0.1):
    valid_ds = np.ones(len(dw))
    for i in range(len(dw)):
        next_state = update_complex_state(state, dw[i], time_step)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

        # print(f"State: {state} + Action: {dw[i]} --> Expected: {next_state}  :: Safe: {safe}")

    return valid_ds 



@njit(cache=True)
def modify_action(valid_window, dw):
    """ 
    By the time that I get here, I have already established that pp action is not ok so I cannot select it, I must modify the action. 
    """
    idx_search = int((len(dw)-1)/2)
    d_size = len(valid_window)
    for i in range(d_size):
        p_d = int(min(d_size-1, idx_search+i))
        if valid_window[p_d]:
            return dw[p_d]
        n_d = int(max(0, idx_search-i))
        if valid_window[n_d]:
            return dw[n_d]

    

class BaseKernel:
    def __init__(self, sim_conf, plotting):
        self.resolution = sim_conf.n_dx
        self.plotting = plotting

    def view_kernel(self, theta):
        phi_range = np.pi
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        plt.figure(5)
        plt.title(f"Kernel phi: {theta} (ind: {theta_ind})")
        img = self.kernel[:, :, theta_ind].T 
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def check_state(self, state=[0, 0, 0]):
        i, j, k= self.get_indices(state)

        # print(f"Expected Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
        if self.plotting:
            self.plot_kernel_point(i, j, k)
        if self.kernel[i, j, k] != 0:
            return False # unsfae state
        return True # safe state

    def plot_kernel_point(self, i, j, k):
        plt.figure(5)
        plt.clf()
        plt.title(f"Kernel inds: {i}, {j}, {k}")
        img = self.kernel[:, :, k].T 
        plt.imshow(img, origin='lower')
        plt.plot(i, j, 'x', markersize=20, color='red')
        # plt.show()
        plt.pause(0.0001)

class ForestKernel(BaseKernel):
    def __init__(self, sim_conf, plotting=False):
        super().__init__(sim_conf, plotting)
        self.kernel = None
        self.side_kernel = np.load(f"{sim_conf.kernel_path}SideKernel_{sim_conf.kernel_name}.npy")
        self.obs_kernel = np.load(f"{sim_conf.kernel_path}ObsKernel_{sim_conf.kernel_name}.npy")
        img_size = int(sim_conf.obs_img_size * sim_conf.n_dx)
        obs_size = int(sim_conf.obs_size * sim_conf.n_dx)
        self.obs_offset = int((img_size - obs_size) / 2)

    def construct_kernel(self, track_size, obs_locations):
        self.kernel = construct_forest_kernel(track_size, obs_locations, self.resolution, self.side_kernel, self.obs_kernel, self.obs_offset)

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.kernel.shape[2]-1)

        return x_ind, y_ind, theta_ind


@njit(cache=True)
def construct_forest_kernel(track_size, obs_locations, resolution, side_kernel, obs_kernel, obs_offset):
    kernel = np.zeros((track_size[0], track_size[1], side_kernel.shape[2]))
    length = int(track_size[1] / resolution)
    for i in range(length):
        kernel[:, i*resolution:(i+1)*resolution] = side_kernel

    if obs_locations is None:
        return kernel

    for obs in obs_locations:
        i = int(round(obs[0] * resolution)) - obs_offset
        j = int(round(obs[1] * resolution)) - obs_offset * 2
        if i < 0:
            kernel[0:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel[abs(i):obs_kernel.shape[0], :]
            continue

        if kernel.shape[0] - i <= (obs_kernel.shape[0]):
            kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel[0:obs_kernel.shape[0]-i, :]
            continue


        kernel[i:i+obs_kernel.shape[0], j:j+obs_kernel.shape[1]] += obs_kernel
    
    return kernel


class TrackKernel(BaseKernel):
    def __init__(self, sim_conf, plotting=False, kernel_name=None):
        super().__init__(sim_conf, plotting)
        if kernel_name is None:
            kernel_name = f"{sim_conf.kernel_path}Kernel_{sim_conf.track_kernel_mode}_{sim_conf.map_name}.npy"
        else:
            kernel_name = f"{sim_conf.kernel_path}{kernel_name}"
        self.clean_kernel = np.load(kernel_name)
        self.kernel = self.clean_kernel.copy()
        self.phi_range = sim_conf.phi_range
        self.n_modes = sim_conf.n_modes
        self.max_steer = sim_conf.max_steer

        # self.obs_kernel = np.load(f"{sim_conf.kernel_path}ObsKernelTrack_{sim_conf.track_kernel_path}.npy")
        img_size = int(sim_conf.obs_img_size * sim_conf.n_dx)
        obs_size = int(sim_conf.obs_size * sim_conf.n_dx)
        self.obs_offset = int((img_size - obs_size) / 2)

        file_name = 'maps/' + sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = yaml_file['origin']

    def construct_kernel(self, a, obs_locations):
        if len(obs_locations) == 0:
            self.kernel = self.clean_kernel
            return

        resize = self.clean_kernel.shape[0] / a[1]
        obs_locations *= resize
        self.kernel = construct_track_kernel(self.clean_kernel, obs_locations, self.obs_kernel, self.obs_offset)

        theta_ind = int(round((0 + self.phi_range/2) / self.phi_range * (self.kernel.shape[2]-1)))
        plt.figure(5)
        plt.title(f"Kernel phi: {0} (ind: {theta_ind}) combined")
        img = self.kernel[:, :, theta_ind].T + self.clean_kernel[:, :, theta_ind].T
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def get_indices(self, state):
        phi_range = np.pi * 2
        x_ind = min(max(0, int(round((state[0]-self.origin[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-self.origin[1])*self.resolution))), self.kernel.shape[1]-1)

        phi = state[2]
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))

        # mode_ind = min(max(0, int(round((state[4]+self.max_steer)*self.n_modes ))), self.kernel.shape[3]-1)

        return x_ind, y_ind, theta_ind


@njit(cache=True)
def construct_track_kernel(clean_kernel, obs_locations, obs_kernel, obs_offset):
    kernel = np.copy(clean_kernel)

    for obs in obs_locations:
        i = int(round(obs[0] )) - obs_offset
        j = int(round(obs[1] )) - obs_offset

        i_start = i 
        i_kernel = 0 
        i_len = obs_kernel.shape[0]
        j_start = j
        j_kernel = 0
        j_len = obs_kernel.shape[1]
        
        if i < 0:
            i_start = 0
            i_kernel = abs(i)
            i_len = obs_kernel.shape[0] - i_kernel

        elif kernel.shape[0] - i <= (obs_kernel.shape[0]):
            i_len = kernel.shape[0] - i

        if j < 0:
            j_start = 0
            j_kernel = abs(j)
            j_len = obs_kernel.shape[1] - j_kernel

        elif kernel.shape[1] - j <= (obs_kernel.shape[1]):
            j_len = kernel.shape[1] - j

        additive = obs_kernel[i_kernel:i_kernel + i_len, j_kernel:j_kernel + j_len, :]
        kernel[i_start:i_start + i_len, j_start:j_start + j_len, :] += additive

    return kernel




