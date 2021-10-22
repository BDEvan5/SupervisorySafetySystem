import numpy as np
from numba import njit
from matplotlib import pyplot as plt


class Kernel:
    def __init__(self, sim_conf):
        self.kernel = None
        self.resolution = sim_conf.n_dx
        self.side_kernel = np.load(f"{sim_conf.kernel_path}SideKernel_{sim_conf.kernel_name}.npy")
        self.obs_kernel = np.load(f"{sim_conf.kernel_path}ObsKernel_{sim_conf.kernel_name}.npy")

    def construct_kernel(self, track_size, obs_locations):
        self.kernel = np.zeros((track_size[0], track_size[1], self.side_kernel.shape[2]))
        length = int(track_size[1] / self.resolution)
        for i in range(length):
            self.kernel[:, i*self.resolution:(i+1)*self.resolution] = self.side_kernel

        offset = [40, 80] #TODO: see issue here
        resolution = 100
        for obs in obs_locations:
            i = int(round(obs[0] * resolution)) - offset[0]
            j = int(round(obs[1] * resolution)) - offset[1]
            if i < 0:
                self.kernel[0:i+self.obs_kernel.shape[0], j:j+self.obs_kernel.shape[1]] += self.obs_kernel[abs(i):self.kernel.shape[0], :]
                continue

            if self.kernel.shape[0] - i <= (self.obs_kernel.shape[0]):
                self.kernel[i:i+self.obs_kernel.shape[0], j:j+self.obs_kernel.shape[1]] += self.obs_kernel[0:self.kernel.shape[0]-i, :]
                continue


            self.kernel[i:i+self.obs_kernel.shape[0], j:j+self.obs_kernel.shape[1]] += self.obs_kernel

        self.kernel = np.clip(self.kernel, 0, 1)

        # self.view_kernel(np.pi/4)

    def view_kernel(self, theta):
        phi_range = np.pi
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        plt.figure(5)
        plt.title(f"Kernel phi: {theta} (ind: {theta_ind})")
        img = self.kernel[:, :, theta_ind].T 
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        theta_ind = min(max(0, theta_ind), self.kernel.shape[2]-1)

        return x_ind, y_ind, theta_ind

    def check_state(self, state=[0, 0, 0]):
        i, j, k = self.get_indices(state)

        # print(f"Expected Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
        # self.plot_kernel_point(i, j, k)
        if self.kernel[i, j, k] == 1:
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



class SafetyWrapper:
    def __init__(self, planner, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        #TODO: make sure these parameters are defined in the planner an then remove them here. This is constructor dependency injection
        self.d_max = 0.4 # radians  
        self.v = 2
        self.kernel = Kernel(conf)
        self.planner = planner

    def plan(self, obs):
        init_action = self.planner.plan_act(obs)
        state = np.array(obs['state'])[0:3]

        safe, next_state = check_init_action(state, init_action, self.kernel)
        if safe:
            return init_action

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            return init_action
        
        action = modify_action(init_action, valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")


        return action

    def generate_dw(self):
        n_segments = 5
        dw = np.ones((5, 2))
        dw[:, 0] = np.linspace(-self.d_max, self.d_max, n_segments)
        dw[:, 1] *= self.v
        return dw



def check_init_action(state, u0, kernel):
    next_state = update_state(state, u0, 0.2)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

def simulate_and_classify(state, dw, kernel):
    valid_ds = np.ones(len(dw))
    for i in range(len(dw)):
        next_state = update_state(state, dw[i], 0.2)
        safe = kernel.check_state(next_state)
        valid_ds[i] = safe 

        # print(f"State: {state} + Action: {dw[i]} --> Expected: {next_state}  :: Safe: {safe}")

    return valid_ds 



# no change required
def modify_action(pp_action, valid_window, dw):
    """ 
    By the time that I get here, I have already established that pp action is not ok so I cannot select it, I must modify the action. 
    """
    d_idx_search = np.argmin(np.abs(dw[:, 0]))
    d_idx = int(find_new_action(valid_window, d_idx_search))
    return dw[d_idx]
    
# no change required
def find_new_action(valid_window, idx_search):
    d_size = len(valid_window)
    for i in range(len(valid_window)):
        p_d = min(d_size-1, idx_search+i)
        if valid_window[p_d]:
            return p_d
        n_d = max(0, idx_search-i)
        if valid_window[n_d]:
            return n_d
    print("No new action: returning only valid via count_nonzero")
    return np.count_nonzero(valid_window)
    

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




