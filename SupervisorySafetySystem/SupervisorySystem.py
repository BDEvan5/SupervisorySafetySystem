from SupervisorySafetySystem.Utils import load_conf
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import yaml, csv
from SupervisorySafetySystem.Simulator.Dynamics import update_complex_state, update_std_state
from SupervisorySafetySystem.Modes import Modes

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
        planned = np.array(self.planned_actions)
        safe = np.array(self.safe_actions)
        plt.figure(5)
        plt.clf()
        plt.title("Safe History: steering")
        plt.plot(planned[:, 0], color='blue')
        plt.plot(safe[:, 0], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        plt.figure(6)
        plt.clf()
        plt.title("Safe History: velocity")
        plt.plot(planned[:, 1], color='blue')
        plt.plot(safe[:, 1], '-x', color='red')
        plt.legend(['Planned Actions', 'Safe Actions'])
        plt.ylim([-0.5, 0.5])
        # plt.show()
        plt.pause(0.0001)

        self.planned_actions = []
        self.safe_actions = []

    def save_safe_history(self, path, name):
        self.plot_safe_history()

        plt.figure(5)
        plt.savefig(f"{path}/{name}_steer_actions.png")

        plt.figure(6)
        plt.savefig(f"{path}/{name}_velocity_actions.png")

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
    def __init__(self, planner, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        self.d_max = conf.max_steer
        self.kernel = TrackKernel(conf)
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

        self.m = Modes(conf)

    def plan(self, obs):
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        safe, next_state = self.check_init_action(state, init_action)
        if safe:
            self.safe_history.add_locations(init_action, init_action)
            return init_action

        # print(f"Intervening")
        valids = self.simulate_and_classify(state)
        if not valids.any():
            print(f"No Valid options -> State: {obs['state']}")
            return init_action
        
        action, idx = modify_mode(valids, self.m.nq_velocity, self.m.nv_modes, self.m.nv_level_modes, self.m.qs)
        self.safe_history.add_locations(init_action, action)

        return action

    def check_init_action(self, state, init_action):
        next_state = update_complex_state(state, init_action, self.time_step/2)
        safe = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.max_steer, self.m.v_mode_list, self.m.nv_modes, self.m.v_res, self.m.min_velocity)
        if not safe:
            return safe, next_state
        next_state = update_complex_state(state, init_action, self.time_step)
        safe = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.max_steer, self.m.v_mode_list, self.m.nv_modes, self.m.v_res, self.m.min_velocity)
        
        return safe, next_state

    def simulate_and_classify(self, state):
        valids = np.ones(len(self.m.qs))
        for i in range(len(self.m.qs)):
            next_state = update_complex_state(state, self.m.qs[i], self.time_step)
            valids[i] = check_kernel_state(next_state, self.kernel.kernel, self.kernel.origin, self.kernel.resolution, self.kernel.phi_range, self.m.max_steer, self.m.v_mode_list, self.m.nv_modes, self.m.v_res, self.m.min_velocity)

        return valids


class LearningSupervisor(Supervisor):
    def __init__(self, planner, conf):
        Supervisor.__init__(self, planner, conf)
        self.intervention_mag = 0
        self.mag_reward = conf.mag_reward
        self.constant_reward = conf.constant_reward
        self.ep_interventions = 0
        self.intervention_list = []
        self.lap_times = []

    def done_entry(self, s_prime, steps=0):
        s_prime['reward'] = self.calculate_reward(self.intervention_mag, s_prime)
        self.planner.done_entry(s_prime)
        self.intervention_list.append(self.ep_interventions)
        self.ep_interventions = 0
        self.lap_times.append(steps)

    def lap_complete(self, steps):
        self.planner.lap_complete()
        self.intervention_list.append(self.ep_interventions)
        self.ep_interventions = 0
        self.lap_times.append(steps)

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

        full_name = self.planner.path + f'/{self.planner.name}_laptime_list.csv'
        data = []
        for i in range(len(self.lap_times)):
            data.append([i, self.lap_times[i]])
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        plt.figure(6)
        plt.clf()
        plt.plot(self.lap_times)
        plt.savefig(f"{self.planner.path}/{self.planner.name}_laptimes.png")
        plt.savefig(f"{self.planner.path}/{self.planner.name}_laptimes.svg")

    def plan(self, obs):
        if abs(self.intervention_mag) > 0:
            obs['reward'] = self.calculate_reward(self.intervention_mag, obs)
            self.planner.intervention_entry(obs)
            init_action = self.planner.plan_act(obs, False)
        else:
            init_action = self.planner.plan_act(obs, True)

        state = np.array(obs['state'])

        # init_mode_action = self.m.action2mode(init_action)
        safe, next_state = self.check_init_action(state, init_action)

        if safe:
            self.intervention_mag = 0
            self.safe_history.add_locations(init_action[0], init_action[0])
            return init_action

        self.ep_interventions += 1
        self.intervene = True

        valids = self.simulate_and_classify(state)
        if not valids.any():
            print(f"No Valid options --> State: {obs['state']}")
            self.intervention_mag = 1
            return init_action

        action, idx = modify_mode(valids, self.m.nq_velocity, self.m.nv_modes, self.m.nv_level_modes, self.m.qs)
        self.safe_history.add_locations(init_action[0], action[0])

        self.intervention_mag = (action[0] - init_action[0])/self.d_max

        return action

    def calculate_reward(self, intervention_mag, obs):
        mag = - self.mag_reward * abs(intervention_mag)
        total_reward = mag - self.constant_reward + obs['reward']
        return  total_reward

@njit(cache=True)
def modify_mode(valid_window, nq_velocity, nv_modes, nv_level_modes, qs):
    """ 
    modifies the action for obstacle avoidance only, it doesn't check the dynamic limits here.
    """
    assert valid_window.any() == 1, "No valid actions:check modify_mode method"

    for vm in range(nq_velocity-1, -1, -1):
        idx_search = int(nv_modes[vm] +(nv_level_modes[vm]-1)/2) 
        if valid_window[idx_search]:
            return qs[idx_search], idx_search

        if nv_level_modes[vm] == 1:
            continue

        d_search_size = int((nv_level_modes[vm]-1)/2)
        for dind in range(d_search_size+1): 
            p_d = int(idx_search+dind)
            if valid_window[p_d]:
                return qs[p_d], p_d
            n_d = int(idx_search-dind-1)
            if valid_window[n_d]:
                return qs[n_d], n_d
        
@njit(cache=True)
def check_state_modes(v, d):
    b = 0.523
    g = 9.81
    l_d = 0.329
    if abs(d) < 0.06:
        return True # safe because steering is small
    friction_v = np.sqrt(b*g*l_d/np.tan(abs(d))) *1.1 # nice for the maths, but a bit wrong for actual friction
    if friction_v > v:
        return True # this is allowed mode
    return False # this is not allowed mode: the friction is too high

@njit(cache=True)
def check_kernel_state(state, kernel, origin, resolution, phi_range, d_max, v_mode_list, nv_modes, v_res, v_min):
        # if not check_state_modes(state[3], state[4]):
        #     return False # unsafe
        #?: this makes it crash, find out why???

        x_ind = min(max(0, int(round((state[0]-origin[0])*resolution))), kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-origin[1])*resolution))), kernel.shape[1]-1)

        phi = state[2]
        if phi >= phi_range/2:
            phi = phi - phi_range
        elif phi < -phi_range/2:
            phi = phi + phi_range
        theta_ind = int(round((phi + phi_range/2) / phi_range * (kernel.shape[2]-1)))

        v_ind = min(max(0, int(round((state[3]-v_min)*v_res))), nv_modes.shape[0]-1)
        d_min, di = 1000, None
        for i in range(len(v_mode_list[v_ind])):
            d_dis = abs(v_mode_list[v_ind][i] - state[4])
            if d_dis < d_min:
                d_min, di = d_dis, i
        d_ind = min(max(0, int(round(di))), v_mode_list[v_ind].shape[0]-1)
        mode = int(nv_modes[v_ind] + d_ind)
        
        if kernel[x_ind, y_ind, theta_ind, mode] != 0:
            return False # unsfae state
        return True # safe state



class TrackKernel:
    def __init__(self, sim_conf, plotting=False, kernel_name=None):
        if kernel_name is None:
            kernel_name = f"{sim_conf.kernel_path}Kernel_{sim_conf.kernel_mode}_{sim_conf.map_name}.npy"
        else:
            kernel_name = f"{sim_conf.kernel_path}{kernel_name}"
        self.kernel = np.load(kernel_name)

        self.plotting = plotting
        self.m = Modes(sim_conf)
        self.resolution = sim_conf.n_dx
        self.phi_range = sim_conf.phi_range
        self.max_steer = sim_conf.max_steer
        self.n_modes = self.m.n_modes

        file_name = 'maps/' + sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = np.array(yaml_file['origin'])

    def check_state(self, state=[0, 0, 0, 0, 0]):
        i, j, k, m = self.get_indices(state)

        if self.plotting:
            self.plot_kernel_point(i, j, k, m)
        if self.kernel[i, j, k, m] != 0:
            return False # unsfae state
        return True # safe state

    def plot_kernel_point(self, i, j, k, m):
        plt.figure(6)
        plt.clf()
        plt.title(f"Kernel inds: {i}, {j}, {k}, {m}: {self.m.qs[m]}")
        img = self.kernel[:, :, k, m].T 
        plt.imshow(img, origin='lower')
        plt.plot(i, j, 'x', markersize=20, color='red')
        plt.pause(0.0001)

    def get_indices(self, state):
        x_ind = min(max(0, int(round((state[0]-self.origin[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1]-self.origin[1])*self.resolution))), self.kernel.shape[1]-1)

        phi = state[2]
        if phi >= self.phi_range/2:
            phi = phi - self.phi_range
        elif phi < -self.phi_range/2:
            phi = phi + self.phi_range
        theta_ind = int(round((phi + self.phi_range/2) / self.phi_range * (self.kernel.shape[2]-1)))
        mode = self.m.get_mode_id(state[3], state[4])

        return x_ind, y_ind, theta_ind, mode


if __name__ == "__main__":
    conf = load_conf("std_test_kernel")
    t = TrackKernel(conf)


