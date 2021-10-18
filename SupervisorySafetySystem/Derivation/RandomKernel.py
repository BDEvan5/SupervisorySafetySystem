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
        
    def reset(self, rand_seed=0):
        np.random.seed(rand_seed)
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


class DiscriminatingImgKernel:
    def __init__(self, track_img):
        self.track_img = track_img
        self.resolution = 100
        self.t_step = 0.2
        self.velocity = 2
        self.n_phi = 61
        self.phi_range = np.pi
        self.half_block = 1 / (2*self.resolution)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = 5

        self.n_x = track_img.shape[0]
        self.n_y = track_img.shape[1]
        self.xs = np.linspace(0, 2, self.n_x)
        self.ys = np.linspace(0, 25, self.n_y)
        self.phis = np.linspace(-self.phi_range/2, self.phi_range/2, self.n_phi)
        
        self.qs = None

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.previous_kernel = np.zeros((self.n_x, self.n_y, self.n_phi))
        self.build_qs()
        self.dynamics = build_dynamics_table(self.phis, self.qs, self.velocity, self.t_step, self.resolution)

        self.kernel[:, :, :] = track_img[:, :, None] * np.ones((self.n_x, self.n_y, self.n_phi))
        self.kernel[0, :, :] = 1
        self.kernel[-1, :, :] = 1
        # self.previous_kernel = np.copy(self.kernel)

    # config functions
    def build_qs(self):
        max_steer = 0.35
        ds = np.linspace(-max_steer, max_steer, self.n_modes)
        self.qs = self.velocity / 0.33 * np.tan(ds)

    def calculate_kernel(self, n_loops=1):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

            plt.figure(2)
            plt.title(f"Kernel after loop: {z}")
            phi_n = 30
            img = self.kernel[:, :, phi_n].T - self.previous_kernel[:, :, phi_n].T
            plt.imshow(img, origin='lower')
            plt.pause(0.0001)

            self.view_kernel(0, False)
        self.save_kernel()

    def save_kernel(self):
        np.save("SupervisorySafetySystem/Discrete/SetObsKern.npy", self.kernel)
        print(f"Saved kernel to file")

    def load_kernel(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/SetObsKern.npy")

    def view_kernel(self, phi, show=True):
        phi_ind = np.argmin(np.abs(self.phis - phi))
        plt.figure(1)
        plt.title(f"Kernel phi: {phi} (ind: {phi_ind})")
        # mode = int((self.n_modes-1)/2)
        mode = 4
        img = self.kernel[:, :, phi_ind].T 
        plt.imshow(img, origin='lower')

        arrow_len = 0.15
        plt.arrow(0, 0, np.sin(phi)*arrow_len, np.cos(phi)*arrow_len, color='r', width=0.001)
        for m in range(self.n_modes):
            i, j = int(self.n_x/2), 0 
            di, dj, new_k = self.dynamics[phi_ind, m, 0,-1]


            plt.arrow(i, j, di, dj, color='b', width=0.001)

        plt.pause(0.0001)
        if show:
            plt.show()

class Kernel:
    def __init__(self):
        self.kernel = np.load("SupervisorySafetySystem/Discrete/SetObsKern.npy")
        self.resolution = 100

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

    def get_indices(self, state):
        phi_range = np.pi
        x_ind = min(max(0, int(round((state[0])*self.resolution))), self.kernel.shape[0]-1)
        y_ind = min(max(0, int(round((state[1])*self.resolution))), self.kernel.shape[1]-1)
        theta_ind = int(round((state[2] + phi_range/2) / phi_range * (self.kernel.shape[2]-1)))

        return x_ind, y_ind, theta_ind


    def check_state(self, state=[0, 0, 0]):
        i, j, k = self.get_indices(state)

        print(f"Location: {state} -> Inds: {i}, {j}, {k} -> Value: {self.kernel[i, j, k]}")
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



class RandoPlanner:
    def __init__(self):
        self.d_max = 0.4 # radians  
        self.v = 2
        # self.history = History()
        self.kernel = Kernel()

    def plan(self, obs):
        # obstacles = generate_cheat_obs(obs, self.d_max)  # purely for plotting
        # self.kernel.add_obstacles(obs)
        # pp_action = self.run_pure_pursuit(obs['state'])
        pp_action = self.take_random_action()
        state = obs['state']

        safe, next_state = check_init_action(state, pp_action, self.kernel)
        if safe:
            self.plot_single_flower(obs, next_state)
            return pp_action

        # sample actions
        dw = self.generate_dw()
        next_states = simulate_sampled_actions(state, dw)
        valids = classify_next_states(next_states, self.kernel)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            print(f"Next_states: {next_states}")
            # if self.history.obstacles is not None:
                # print(f"Previous action: {self.history.action[0]}")
                # self.plot_local_linky(self.history.obstacles, self.history.observation, self.history.valids, self.history.next_states, 4)
            # self.plot_local_linky(obstacles, obs, valids, next_states, 3)
            # self.plot_flower(obs, next_states, obstacles, valids)
            # plt.show()
            return pp_action
        
        action = modify_action(pp_action, valids, dw)
        print(f"Valids: {valids} -> new action: {action}")

        # self.plot_flower(obs, next_states, obstacles, valids)
        # print(f"Action mod>> o:{pp_action[0]} --> n:{action[0]}")

        # self.plot_local_linky(obstacles, obs, valids, next_states, 3)
        # self.history.add_data(obstacles, obs, valids, next_states, action)
        # plt.show()

        return action

    def take_random_action(self):
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        return np.array([steering, self.v])


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

    def plot_single_flower(self, observation, next_state, obstacles=None):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: ')

        plt.ylim([0, 3])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(observation['full_scan'])
        plt.plot(xs, ys, '-+')

        if obstacles is not None:
            for obs in obstacles:
                obs.plot_obstacle()
            
        x_p = [0, next_state[0]]
        y_p = [0, next_state[1]]
        plt.plot(x_p, y_p, '--', color='green')

        plt.pause(0.0001)

    def plot_local_linky(self, observation, valids, next_states, figure_n=2, obstacles=None):
        plt.figure(figure_n)
        plt.clf()
        plt.title(f'Lidar Scan: ')

        plt.ylim([-0.5, 1.0])
        plt.xlim([-0.75, 0.75])


        if obstacles is not None:
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


# def generate_cheat_obs(obs, d_max):
#     pts1 = obs['obs_pts1']
#     pts2 = obs['obs_pts2']
    
#     obses = []
#     for pt1, pt2 in zip(pts1, pts2):
#         obs = ObstacleThree(pt1, pt2, d_max, len(obses))
#         obses.append(obs)
    
#     return obses
    

def check_init_action(state, u0, kernel):
    # state = np.array([0, 0, 0])
    next_state = update_state(state, u0, 0.1)
    safe = kernel.check_state(next_state)
    
    return safe, next_state

# no changes required 
def simulate_sampled_actions(state, dw):
    # state = np.array([0, 0, 0])
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
    

  


# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, resolution):
    # add 5 sample points
    block_size = 1 / (resolution)
    h = 1 * block_size
    phi_size = np.pi / (len(phis) -1)
    ph = 0.1 * phi_size
    n_pts = 5
    dynamics = np.zeros((len(phis), len(qs), n_pts, 8, 3), dtype=np.int)
    phi_range = np.pi
    n_steps = 1
    for i, p in enumerate(phis):
        for j, m in enumerate(qs):
            for t in range(n_pts):
                t_step = time * (t+1)  / n_pts
                phi = p + m * t_step * n_steps # phi must be at end
                dx = np.sin(phi) * velocity * t_step
                dy = np.cos(phi) * velocity * t_step
                
                new_k_min = int(round((phi - ph + phi_range/2) / phi_range * (len(phis)-1)))
                new_k_max = int(round((phi + ph + phi_range/2) / phi_range * (len(phis)-1)))
                dynamics[i, j, t, 0:4, 2] = min(max(0, new_k_min), len(phis)-1)
                dynamics[i, j, t, 4:8, 2] = min(max(0, new_k_max), len(phis)-1)

                dynamics[i, j, t, 0, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 0, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, t, 1, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 1, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, t, 2, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 2, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, t, 3, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 3, 1] = int(round((dy -h) * resolution))

                dynamics[i, j, t, 4, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 4, 1] = int(round((dy -h) * resolution))
                dynamics[i, j, t, 5, 0] = int(round((dx -h) * resolution))
                dynamics[i, j, t, 5, 1] = int(round((dy +h) * resolution))
                dynamics[i, j, t, 6, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 6, 1] = int(round((dy +h )* resolution))
                dynamics[i, j, t, 7, 0] = int(round((dx +h) * resolution))
                dynamics[i, j, t, 7, 1] = int(round((dy -h) * resolution))

                pass

    return dynamics

# @jit(cache=True)
def kernel_loop(kernel, xs, ys, phis, n_modes, dynamics):
    previous_kernel = np.copy(kernel)
    for i in range(len(xs)):
        for j in range(len(ys)):
            for k in range(len(phis)):
                    if kernel[i, j, k] == 1:
                        continue 
                    kernel[i, j, k] = check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel, xs, ys)

    return kernel

@njit(cache=True)
def check_kernel_state(i, j, k, n_modes, dynamics, previous_kernel, xs, ys):
    n_pts = 5
    for l in range(n_modes):
        safe = True
        # check all concatanation points and offsets and if none are occupied, then it is safe.
        for t in range(n_pts):
            for n in range(dynamics.shape[3]):
                di, dj, new_k = dynamics[k, l, t, n, :]
                new_i = min(max(0, i + di), len(xs)-1)  
                new_j = min(max(0, j + dj), len(ys)-1)

                if previous_kernel[new_i, new_j, new_k]:
                    # if you hit a constraint, break
                    safe = False # breached a limit.
                    break

            # if not previous_kernel[new_i, new_j, new_k] and t == n_pts - 1:
            #     return False
        if safe:
            return False

    return True








if __name__ == "__main__":
    env = SimThree()  
    env.reset()

    # kernel = DiscriminatingImgKernel(env.env_map.map_img)
    # kernel.calculate_kernel(20)
    # kernel.view_kernel(0, True)


    planner = RandoPlanner()
    success = 0

    for i in range(1):
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

        # if r == -1:
        plt.show()

    print("Success rate: {}".format(success/100))

