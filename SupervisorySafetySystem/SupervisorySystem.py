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

        self.m = Modes(conf)

    def plan(self, obs):
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        init_mode_action = self.action2mode(init_action)
        safe, next_state = self.check_init_action(state, init_mode_action)
        if safe:
            self.safe_history.add_locations(init_mode_action[0], init_mode_action[0])
            return init_action

        valids = simulate_and_classify(state, self.m.qs, self.kernel, self.time_step)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            # plt.show()
            return init_action
        
        action = modify_mode(self.m, valids)
        self.safe_history.add_locations(init_action[0], action[0])

        return action

    def action2mode(self, init_action):
        id = self.m.get_mode_id(init_action[1], init_action[0])
        return self.m.qs[id]

    def check_init_action(self, state, init_action):
        d, v = init_action
        b = 0.523
        g = 9.81
        l_d = 0.329
        if abs(d)> 0.06: 
            #  only check the friction limit if it might be a problem
            friction_v = np.sqrt(b*g*l_d/np.tan(abs(d))) *1.1
            if friction_v < v:
                print(f"Invalid action: check planner or expect bad resultsL {init_action} -> max_friction_v: {friction_v}")
                return False, state

        next_state = update_complex_state(state, init_action, self.time_step)
        safe = self.kernel.check_state(next_state)
        
        return safe, next_state

class LearningSupervisor(Supervisor):
    def __init__(self, planner, kernel, conf):
        Supervisor.__init__(self, planner, kernel, conf)
        self.intervention_mag = 0
        self.calculate_reward = None # to be replaced by a function
        self.ep_interventions = 0
        self.intervention_list = []
        self.lap_times = []

    def done_entry(self, s_prime, steps=0):
        s_prime['reward'] = self.calculate_reward(self.intervention_mag, s_prime)
        self.planner.done_entry(s_prime)
        self.intervention_list.append(self.ep_interventions)
        self.ep_interventions = 0
        self.lap_times.append(steps)

    def fake_done(self, steps):
        self.planner.fake_done()
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
        obs['reward'] = self.calculate_reward(self.intervention_mag, obs)
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])

        fake_done = False
        if abs(self.intervention_mag) > 0: fake_done = True

        init_mode_action = self.action2mode(init_action)
        safe, next_state = self.check_init_action(state, init_mode_action)

        if safe:
            self.intervention_mag = 0
            self.safe_history.add_locations(init_action[0], init_action[0])
            return init_mode_action, fake_done

        self.ep_interventions += 1
        self.intervene = True

        valids = simulate_and_classify(state, self.m.qs, self.kernel, self.time_step)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            # plt.show()
            self.intervention_mag = 1
            return init_action, fake_done

        action = modify_mode(self.m, valids)
        self.safe_history.add_locations(init_action[0], action[0])

        self.intervention_mag = (action[0] - init_action[0])/self.d_max

        return action, fake_done


def simulate_and_classify(state, dw, kernel, time_step):
    valid_ds = np.ones(len(dw))
    for i in range(len(dw)):
        next_state = update_complex_state(state, dw[i], time_step)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

        # print(f"State: {state} + Action: {dw[i]} --> Expected: {next_state}  :: Safe: {safe}")

    return valid_ds 

#TODO: JIT this
def modify_mode(self: Modes, valid_window):
    """ 
    modifies the action for obstacle avoidance only, it doesn't check the dynamic limits here.
    """
    # max_v_idx = 
    #TODO: decrease the upper limit of the search according to the velocity
    for vm in range(self.nq_velocity-1, 0, -1):
        idx_search = int(self.nv_modes[vm] +(self.nv_level_modes[vm]-1)/2) # idx to start searching at.

        if self.nv_level_modes[vm] == 1:
            if valid_window[idx_search]:
                # if idx_search == 8 or idx_search == 9:
                    # print(f"Mode idx: {idx_search} -> {self.qs[idx_search]}")
                return self.qs[idx_search]
            continue

        # at this point there are at least 3 steer options
        d_search_size = int((self.nv_level_modes[vm]-1)/2)
        for dind in range(d_search_size+1): # for d_ss=1 it should search, 0 and 1.
            p_d = int(idx_search+dind)
            if valid_window[p_d]:
                return self.qs[p_d]
            n_d = int(idx_search-dind)
            if valid_window[n_d]:
                return self.qs[n_d]
        

    idx_search = int((len(self.qs)-1)/2)
    d_size = len(valid_window)
    for i in range(d_size):
        p_d = int(min(d_size-1, idx_search+i))
        if valid_window[p_d]:
            return self.qs[p_d]
        n_d = int(max(0, idx_search-i))
        if valid_window[n_d]:
            return self.qs[n_d]


    

class BaseKernel:
    def __init__(self, sim_conf, plotting):
        self.resolution = sim_conf.n_dx
        self.plotting = plotting
        self.m = Modes(sim_conf)

    def view_kernel(self, theta):
        phi_range = np.pi
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        plt.figure(6)
        plt.title(f"Kernel phi: {theta} (ind: {theta_ind})")
        img = self.kernel[:, :, theta_ind].T 
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def check_state(self, state=[0, 0, 0, 0, 0]):
        i, j, k, m = self.get_indices(state)

        # print(f"Expected Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
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
        # plt.show()
        plt.pause(0.0001)

    def print_kernel_area(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")



class TrackKernel(BaseKernel):
    def __init__(self, sim_conf, plotting=False, kernel_name=None):
        super().__init__(sim_conf, plotting)
        if kernel_name is None:
            kernel_name = f"{sim_conf.kernel_path}Kernel_{sim_conf.kernel_mode}_{sim_conf.map_name}.npy"
        else:
            kernel_name = f"{sim_conf.kernel_path}{kernel_name}"
        self.clean_kernel = np.load(kernel_name)
        self.kernel = self.clean_kernel.copy()
        print(f"non: {np.count_nonzero(self.kernel[:, :, :, 8])}")
        print(f"zero: {np.where(self.kernel[:, :, :, 8]==0)}")
        self.phi_range = sim_conf.phi_range
        self.n_modes = self.m.n_modes
        self.max_steer = sim_conf.max_steer

        file_name = 'maps/' + sim_conf.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())
        self.origin = yaml_file['origin']

    def construct_kernel(self, a, obs_locations):
        pass

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
        mode = self.m.get_mode_id(state[3], state[4])

        # mode_ind = min(max(0, int(round((state[4]+self.max_steer)*self.n_modes ))), self.kernel.shape[3]-1)

        return x_ind, y_ind, theta_ind, mode


if __name__ == "__main__":
    conf = load_conf("std_test_kernel")
    t = TrackKernel(conf)


