
import numpy as np
from numba import njit

class Supervisor:
    def __init__(self, planner, kernel, conf):
        """
        A wrapper class that can be used with any other planner.
        Requires a planner with:
            - a method called 'plan_act' that takes a state and returns an action

        """
        
        #TODO: make sure these parameters are defined in the planner an then remove them here. This is constructor dependency injection
        self.d_max = conf.max_steer
        self.v = 2
        # self.kernel = ForestKernel(conf)
        self.kernel = kernel
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

#TODO jit all of this.

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




