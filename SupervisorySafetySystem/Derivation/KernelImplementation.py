import numpy as np
import matplotlib.pyplot as plt

from SupervisorySafetySystem.Derivation.BaseDerivationSim import BaseSim
from numba import njit
from SupervisorySafetySystem.SafetySys.safety_utils import *


class SimThree(BaseSim):
    def __init__(self):
        BaseSim.__init__(self)
        self.state = np.zeros(3) #[x, y, th]

    def update_state(self, action, dt):
        return update_state(self.state, action, dt)
        
    def reset(self):
        self.state = self.env_map.start_pose[0:3]
        return self.base_reset()

    def check_done(self):
        return self.base_check_done()

# @njit(cache=True)
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


class ObstacleThree:
    L = 0.33

    def __init__(self, p1, p2, d_max=0.4, n=0):
        b = 0.05 
        self.op1 = p1 + [-b, -b]
        self.op2 = p2 + [b, -b]
        self.p1 = p1 + [-b, -b]
        self.p2 = p2 + [b, -b]
        self.d_max = d_max * 1
        self.obs_n = n
        self.m_pt = np.mean([self.op1, self.op2], axis=0)
        
    def transform_obstacle(self, theta):
        """
        Calculate transformed points based on theta by constructing rotation matrix.
        """
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        self.p1 = rot_m @ (self.op1 - self.m_pt) + self.m_pt
        self.p2 = rot_m @ (self.op2 - self.m_pt) + self.m_pt

    def transform_point(self, pt, theta):
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])

        relative_pt = pt - self.m_pt
        new_pt = rot_m @ relative_pt
        new_pt += self.m_pt
        return new_pt

    def find_critical_distances(self, state_point_x):
        if state_point_x < self.p1[0] or state_point_x > self.p2[0]:
            return 1, 1 #TODO: think about what values to put here

        w1 = state_point_x - self.p1[0] 
        w2 = self.p2[0] - state_point_x 

        width_thresh = self.L / np.tan(self.d_max)

        d1 = np.sqrt(2*self.L* w1 / np.tan(self.d_max) - w1**2) if w1 < width_thresh else width_thresh
        d2 = np.sqrt(2*self.L * w2 / np.tan(self.d_max) - w2**2) if w2 < width_thresh else width_thresh

        return d1, d2
  
    def calculate_required_y(self, x_value):
        d1, d2 = self.find_critical_distances(x_value)
        corrosponding_y = np.interp(x_value, [self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])

        y = min(self.p1[1], self.p2[1])
        y_safe = max(y - d1, y-d2)

        # y1 = corrosponding_y - d1
        # y2 = corrosponding_y - d2
        # y1 = np.mean([corrosponding_y, self.p1[1]]) - d1
        # y2 = np.mean([corrosponding_y, self.p2[1]]) - d2

        # y_safe, d_star = y1, d1
        # if y1 < y2:
        #     y_safe, d_star = y2, d2

        return y_safe

    def plot_obstacle(self):
        pts = np.vstack((self.p1, self.p2))
        plt.plot(pts[:, 0], pts[:, 1], 'x-', markersize=10, color='black')
        pts = np.vstack((self.op1, self.op2))
        plt.plot(pts[:, 0], pts[:, 1], '--', markersize=20, color='black')

    def calculate_safety(self, state=[0, 0, 0]):
        theta = state[2]
        self.transform_obstacle(theta)

        x_search = np.copy(state[0])
        y_safe = self.calculate_required_y(x_search)
        x, y = self.transform_point([x_search, y_safe], -theta)

        print(f"OrigX: {state[0]} -> SearchX: {x_search} -> newX: {x} -> y safe: {y} -> remaining diff: {state[0] - x}")

        while abs(state[0] - x) > 0.01:
            if x_search == self.p1[0] or x_search == self.p2[0]-0.05:
                print(f"Breakin since x_search: {x_search} ->")
                break
            
            x_search = x_search + (state[0]-x)
            x_search = np.clip(x_search, self.p1[0], self.p2[0]-0.05) #TODO check this end condition
            y_safe = self.calculate_required_y(x_search)
            x, y = self.transform_point([x_search, y_safe], -theta)

            print(f"OrigX: {state[0]} -> SearchX: {x_search} -> newX: {x} -> y safe: {y} -> remaining diff: {state[0] - x}")

        return x, y 

    def run_check(self, state):
        x, y = self.calculate_safety(state)

        state_x = state[0]
        if state_x < self.p1[0] or state_x > self.p2[0]:

            return True 
        if state[1] > self.p1[1] and state[1] > self.p2[1]:
            return False

        if y > state[1]:
            return True 
        return False

    def plot_region(self, theta=0):
        self.transform_obstacle(-0.4)
        xs = np.linspace(self.p1[0], self.p2[0], 20)
        ys = np.array([self.calculate_required_y(x) for x in xs])

        plt.plot(xs, ys, '--', markersize=20, color='black')

        self.transform_obstacle(0)
        xs = np.linspace(self.p1[0], self.p2[0], 20)
        ys = np.array([self.calculate_required_y(x) for x in xs])

        plt.plot(xs, ys, '--', markersize=20, color='black')

        self.transform_obstacle(0.4)
        xs = np.linspace(self.p1[0], self.p2[0], 20)
        ys = np.array([self.calculate_required_y(x) for x in xs])

        plt.plot(xs, ys, '--', markersize=20, color='black')


class History:
    def __init__(self):
        self.obstacles = None 
        self.observation=None #
        self.valids = None
        self.next_states = None
        self.action = None #

    def add_data(self, obstacles, observation, valids, next_states, action):
        self.obstacles = obstacles
        self.observation = observation
        self.valids = valids
        self.next_states = next_states
        self.action = action

class Kernel:
    def __init__(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/ObsKernal_ijk.npy")
        self.obs_pts1 = None 
        self.obs_pts2 = None

        self.resolution = 200
        self.x_offset = -0.25
        self.y_offset = -1.5
        self.offset = np.array([self.x_offset, self.y_offset])

        self.view_kernel(0)

    def view_kernel(self, theta):
        phi_range = np.pi
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))
        plt.figure(5)
        plt.title(f"Kernel phi: {theta} (ind: {theta_ind})")
        img = self.kernel[:, :, theta_ind].T 
        plt.imshow(img, origin='lower')

        # plt.show()
        plt.pause(0.0001)

    def add_obstacles(self, obs):
        self.obs_pts1 = np.array(obs['obs_pts1'])
        self.obs_pts2 = np.array(obs['obs_pts2'])

    def check_state(self, state=[0, 0, 0]):
        for o1, o2 in zip(self.obs_pts1, self.obs_pts2):
            location = state[0:2] - o1 
            if location[1] < self.y_offset:
                continue
            i, j, k = self.get_indices(location, state[2])

            print(f"Location: {location} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
            self.plot_kernel_point(i, j, k)
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

    def get_indices(self, location, theta):
        phi_range = np.pi
        x_ind = min(max(0, int(round((location[0]-self.x_offset)*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((location[1]-self.y_offset)*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((theta + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))

        return x_ind, y_ind, theta_ind




class SafetySystemThree:
    def __init__(self):
        self.d_max = 0.4 # radians  
        self.v = 2
        self.history = History()
        self.kernel = Kernel()

    def plan(self, obs):
        obstacles = generate_cheat_obs(obs, self.d_max)  # purely for plotting
        self.kernel.add_obstacles(obs)
        pp_action = self.run_pure_pursuit(obs['state'])

        safe, next_state = check_init_action(pp_action, self.kernel)
        if safe:
            self.plot_single_flower(obs, next_state, obstacles)
            return pp_action

        # sample actions
        dw = self.generate_dw()
        next_states = simulate_sampled_actions(dw)
        valids = classify_next_states(next_states, self.kernel)
        if not valids.any():
            print('No Valid options')
            if self.history.obstacles is not None:
                print(f"Previous action: {self.history.action[0]}")
                # self.plot_local_linky(self.history.obstacles, self.history.observation, self.history.valids, self.history.next_states, 4)
            # self.plot_local_linky(obstacles, obs, valids, next_states, 3)
            # self.plot_flower(obs, next_states, obstacles, valids)
            plt.show()
            return pp_action
        
        action = modify_action(pp_action, valids, dw)
        print(f"Valids: {valids} -> new action: {action}")

        # self.plot_flower(obs, next_states, obstacles, valids)
        # print(f"Action mod>> o:{pp_action[0]} --> n:{action[0]}")

        # self.plot_local_linky(obstacles, obs, valids, next_states, 3)
        self.history.add_data(obstacles, obs, valids, next_states, action)
        # plt.show()

        return action

    def run_pure_pursuit(self, state):
        lookahead_distance = 1
        L = 0.33
        pose_theta = state[2]
        lookahead = np.array([1, state[1]+lookahead_distance]) #pt 1 m in the future on centerline
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead[0:2]-state[0:2])
        if np.abs(waypoint_y) < 1e-6:
            return np.array([0, self.v])
        radius = 1/(2.0*waypoint_y/lookahead_distance**2)
        steering_angle = np.arctan(L/radius)
        steering_angle = np.clip(steering_angle, -self.d_max, self.d_max)
        return np.array([steering_angle, self.v])

    def generate_dw(self):
        n_segments = 5
        dw = np.ones((5, 2))
        dw[:, 0] = np.linspace(-self.d_max, self.d_max, n_segments)
        dw[:, 1] *= self.v
        return dw

    def plot_flower(self, observation, next_states, obstacles, valids):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: ')

        plt.ylim([0, 3])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(observation['full_scan'])
        plt.plot(xs, ys, '-+')

        for obs in obstacles:
            obs.plot_obstacle(observation['state'])
        
        for i, state in enumerate(next_states):
            x_p = [0, state[0]]
            y_p = [0, state[1]]
            if valids[i]:
                plt.plot(x_p, y_p, '--', color='green')
            else:
                plt.plot(x_p, y_p, '--', color='red')


        plt.pause(0.0001)

    def plot_single_flower(self, observation, next_state, obstacles):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: ')

        plt.ylim([0, 3])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(observation['full_scan'])
        plt.plot(xs, ys, '-+')

        for obs in obstacles:
            obs.plot_obstacle()
        
        x_p = [0, next_state[0]]
        y_p = [0, next_state[1]]
        plt.plot(x_p, y_p, '--', color='green')

        plt.pause(0.0001)

    def plot_local_linky(self, obstacles, observation, valids, next_states, figure_n=2):
        plt.figure(figure_n)
        plt.clf()
        plt.title(f'Lidar Scan: ')

        plt.ylim([-0.5, 1.0])
        plt.xlim([-0.75, 0.75])

        for obs in obstacles:
            obs.plot_obstacle()
            obs.plot_region()

        scale = 0.2        
        for i, state in enumerate(next_states):
            x_p = [0, state[0]]
            y_p = [0, state[1]]
            plt.plot(x_p, y_p, '--', color='purple')
            if valids[i]:
                plt.arrow(state[0], state[1], scale*np.sin(state[2]), scale*np.cos(state[2]), head_width=0.05, head_length=0.1, fc='green', ec='k')
            else:
                plt.arrow(state[0], state[1], scale*np.sin(state[2]), scale*np.cos(state[2]), head_width=0.05, head_length=0.1, fc='red', ec='k')

        # print(valids)
        plt.pause(0.0001)


def generate_cheat_obs(obs, d_max):
    pts1 = obs['obs_pts1']
    pts2 = obs['obs_pts2']
    
    obses = []
    for pt1, pt2 in zip(pts1, pts2):
        obs = ObstacleThree(pt1, pt2, d_max, len(obses))
        obses.append(obs)
    
    return obses
    

def check_init_action(u0, kernel):
    state = np.array([0, 0, 0])
    next_state = update_state(state, u0, 0.1)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

# no changes required 
def simulate_sampled_actions(dw):
    state = np.array([0, 0, 0])
    next_states = np.zeros((len(dw), 3))
    for i in range(len(dw)):
        next_states[i] = update_state(state, dw[i], 0.2)

    return next_states

# no change required
def classify_next_states(next_states, kernel):
    n = len(next_states) 
    valid_ds = np.ones(n)
    for i in range(n):
        safe = kernel.check_state(next_states[i])
        valid_ds[i] = safe 

    return valid_ds 

# no change required
def modify_action(pp_action, valid_window, dw):
    dw_d = dw[:, 0]
    d_idx = np.count_nonzero(dw_d[dw_d<pp_action[0]])
    if valid_window[d_idx]:
        return pp_action
    else:
        d_idx_search = np.argmin(np.abs(dw_d))
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
    

  

if __name__ == "__main__":
    env = SimThree()  
    planner = SafetySystemThree()
    success = 0

    for i in range(100):
        done = False
        state = env.reset()
        while not done:
            a = planner.plan(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render_pose()

        if r == -1:
            print(f"{i}: Crashed")
        elif r == 1:
            print(f"{i}: Success")
            success += 1 

        env.render_ep()

        if r == -1:
            plt.show()

    print("Success rate: {}".format(success/100))



