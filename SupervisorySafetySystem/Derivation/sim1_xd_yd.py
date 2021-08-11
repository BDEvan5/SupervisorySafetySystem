import numpy as np
import matplotlib.pyplot as plt

from SupervisorySafetySystem.Derivation.BaseDerivationSim import BaseSim
from SupervisorySafetySystem.SafetySys.LidarProcessing import segment_lidar_scan, test_fft
from numba import njit
from SupervisorySafetySystem.SafetySys.safety_utils import *


class SimOne(BaseSim):
    def __init__(self):
        BaseSim.__init__(self)
        self.state = np.zeros(2) #[x, y, _]

    def update_state(self, action, dt):
        return update_state(self.state, action, dt)
        
    def reset(self):
        self.state = self.env_map.start_pose[0:2]
        return self.base_reset()

    def check_done(self):
        return self.base_check_done()

# @njit(cache=True)
def update_state(state, action, dt):
    """
    Updates x, y pos accoridng to xd, yd
    """
    dx = np.array(action)
    return state + dx * dt 


class ObstacleOne:
    def __init__(self, p1, p2, xy_ratio, n):
        b = 0.05 
        self.p1 = p1 + [-b, -b]
        self.p2 = p2 + [b, -b]
        self.xy_ratio = xy_ratio * 0.85
        self.obs_n = n

    def run_check(self, state):
        pt = state[0:2]
        o_pt = np.zeros(2)

        # note all gradients are taken from the zero point.
        m1 = self.p1[1] / self.p1[0] 
        m2 = self.p2[1] / self.p2[0] 
        
        if pt[0] < self.p1[0] or pt[0] > self.p2[0]:
            # add code to check if it crosses a line

            # checks if the obs covers the zero mark
            if self.p1[0] < 0 and self.p2[0] > 0:

                # check if thee is the option
                if pt[0] < self.p1[0] and self.p1[0] < 0:
                    # left cross over 
                    m = pt[1] / pt[0]
                    if m < m1:
                        return False
                
                if pt[0] < self.p2[0] and self.p2[0] > 0:
                    # right cross over 
                    m = pt[1] / pt[0]
                    if m > m2:
                        return False     

            return True 
        if pt[1] > self.p1[1] and pt[1] > self.p2[1]:
            return False

        y_required = self.find_critical_point(pt[0])

        if y_required > pt[1]:
            safe_value = True 
        else:
            safe_value = False

        # print(f"{safe_value}: Obs{self.obs_n} -> y_req:{y_required:.4f}, NewPt: {pt} ->start:{self.p1}, end: {self.p2}")

        return safe_value

    def plot_obstacle(self):
        pts = np.vstack((self.p1, self.p2))
        plt.plot(pts[:, 0], pts[:, 1], 'x-', markersize=20)

        xs = np.linspace(self.p1[0], self.p2[0], 10)
        ys = [self.find_critical_point(x) for x in xs]
        plt.plot(xs, ys)

    def find_critical_point(self, x):
        y1 = self.p1[1] - (x - self.p1[0]) / self.xy_ratio
        y2 = self.p2[1] -  (self.p2[0] - x) / self.xy_ratio
        y_safe = max(y1, y2)
        return y_safe 
  

class SafetySystemOne:
    def __init__(self):
        self.xd_lim = 1 
        self.yd = 3
        self.prev_pt = np.zeros(2)
        self.prev_proj_state = np.zeros(2)

    def plan(self, obs):
        xy_ratio = self.xd_lim / self.yd
        obstacles = generate_cheat_obs(obs, xy_ratio)
        pp_action = np.array([0, self.yd])

        # check o action
        safe, next_state = check_init_action(pp_action, obstacles)
        if safe:
            # self.plot_single_flower(obs, next_state, obstacles)
            self.prev_pt = obs['state'][0:2]
            self.prev_proj_state = next_state
            return pp_action


        dw = np.ones((10, 2))
        dw[:, 0] = np.linspace(-self.xd_lim, self.xd_lim, 10)
        dw[:, 1] *= self.yd 

        next_states = simulate_sampled_actions(dw)
        valids = classify_next_states(next_states, obstacles)
        
        if not valids.any():
            print('No Valid options')
            self.plot_flower(obs, next_states, obstacles, valids)
            plt.show()
            return pp_action
        
        action, d_idx = modify_action(pp_action, valids, dw)

        self.plot_flower(obs, next_states, obstacles, valids)
        self.prev_proj_state = next_states[d_idx]
        self.prev_pt = obs['state'][0:2]
        # print(f"Action: {action}")

        return action

    def plot_flower(self, obs, next_states, obstacles, valids):
        plt.figure(2)
        plt.clf()
        plt.title(f'Full Flower: ')

        plt.ylim([-0.5, 2.5])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(obs['full_scan'])
        plt.plot(xs, ys, '-+')

        for obst in obstacles:
            obst.plot_obstacle()
        
        for i, state in enumerate(next_states):
            x_p = [0, state[0]]
            y_p = [0, state[1]]
            if valids[i]:
                plt.plot(x_p, y_p, '--', color='green')
            else:
                plt.plot(x_p, y_p, '--', color='red')

        neg_pt = self.prev_pt - obs['state'][0:2] 
        plt.plot(neg_pt[0], neg_pt[1], 'o', color='black')
        proj = neg_pt + self.prev_proj_state
        plt.plot(proj[0], proj[1], 'o', color='blue', markersize=10)

        plt.pause(0.0001)

    def plot_single_flower(self, obs, next_state, obstacles):
        plt.figure(2)
        plt.clf()
        plt.title(f'Single Flower ')

        plt.ylim([-0.5, 2.5])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(obs['full_scan'])
        plt.plot(xs, ys, '-+')

        for obst in obstacles:
            obst.plot_obstacle()
        
        x_p = [0, next_state[0]]
        y_p = [0, next_state[1]]
        plt.plot(x_p, y_p, '--', color='green')

        neg_pt = self.prev_pt - obs['state'][0:2] 
        plt.plot(neg_pt[0], neg_pt[1], 'o', color='black', markersize=10)
        proj = neg_pt + self.prev_proj_state
        plt.plot(proj[0], proj[1], 'o', color='blue', markersize=10)

        plt.pause(0.0001)


# check the pp action
def check_init_action(u0, obstacles):
    state = np.array([0, 0])
    next_state = update_state(state, u0, 0.1)
    safe = True
    for obs in obstacles:
        if not obs.run_check(next_state):
            safe = False 
            break 
    return safe, next_state
        


def simulate_sampled_actions(dw):
    state = np.zeros(2)
    next_states = np.zeros((len(dw), 2))
    for i in range(len(dw)):
        next_states[i] = update_state(state, dw[i], 0.1)

    return next_states

def generate_cheat_obs(obs, xy_ratio):
    pts1 = obs['obs_pts1']
    pts2 = obs['obs_pts2']
    
    obses = []
    for pt1, pt2 in zip(pts1, pts2):
        obs = ObstacleOne(pt1, pt2, xy_ratio, len(obses))
        obses.append(obs)
    
    return obses
    

def generate_obses(scan, xy_ratio):
    xs, ys = segment_lidar_scan(scan)
    scan_pts = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
    d_cone = 2 # size to consider an obstacle

    new_scan = (xs**2+ys**2)**0.5

    obses = []
    for i in range(len(new_scan)-1):
        pt1 = scan_pts[i]
        pt2 = scan_pts[i+1]
        if new_scan[i] > d_cone or new_scan[i+1] > d_cone:
            continue
        x_lim = 0.5
        if pt1[0] < -x_lim and pt2[0] < -x_lim:
            continue
        if pt1[0] > x_lim and pt2[0] > x_lim:
            continue
        y_lim = 3
        if pt1[1] > y_lim and pt2[1] > y_lim:
            continue
        if i == 0 or i == len(new_scan)-1:
            continue # exclude first and last lines

        if pt1[0] > pt2[0]:
            continue # then the start is after the end, the line slants backwards and isn't to be considered
        
        if new_scan[i] > d_cone:
            f_reduction = d_cone /new_scan[i]
            pt1 = pt1 * f_reduction
        if new_scan[i+1] > d_cone:
            f_reduction = d_cone /new_scan[i+1]
            pt2 = pt2 * f_reduction

        obs = ObstacleOne(pt1, pt2, xy_ratio, len(obses))
        obses.append(obs)

    return obses
    
def classify_next_states(next_states, obstacles):
    n = len(next_states) 
    valid_ds = np.ones(n)
    for i in range(n):
        safe = True 
        for obs in obstacles:
            if not obs.run_check(next_states[i]):
                safe = False 
                break 
        valid_ds[i] = safe 

    return valid_ds 

def modify_action(pp_action, valid_window, dw):
    dw_d = dw[:, 0]
    d_idx = np.count_nonzero(dw_d[dw_d<pp_action[0]])
    if valid_window[d_idx]:
        return pp_action, d_idx
    else:
        d_idx_search = np.argmin(np.abs(dw_d))
        d_idx = int(find_new_action(valid_window, d_idx_search))
        return dw[d_idx], d_idx
    
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
    

  

if __name__ == "__main__":
    env = SimOne()  
    planner = SafetySystemOne()
    success = 0

    for i in range(100):
        done = False
        state = env.reset()
        while not done:
            a = planner.plan(state)
            s_p, r, done, _ = env.step(a)
            state = s_p

        if r == -1:
            print(f"{i}: Crashed")
        elif r == 1:
            print(f"{i}: Success")
            success += 1 

        env.render_ep()

        if r == -1:
            plt.show()

    print("Success rate: {}".format(success/100))

        
    

