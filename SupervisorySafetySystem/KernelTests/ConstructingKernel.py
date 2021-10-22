import numpy as np
import matplotlib.pyplot as plt

from SupervisorySafetySystem.KernelTests.BaseDerivationSim import BaseSim
from numba import njit

#TODO: recombine with proper simulator files
class KernelSim(BaseSim):  #time to move back to proper sim?
    def __init__(self, conf):
        BaseSim.__init__(self, conf)
        self.state = np.zeros(3) #[x, y, th]

    def update_state(self, action, dt):
        return update_state(self.state, action, dt)
        
    def reset(self, rand_seed=0):
        # np.random.seed(rand_seed)
        self.state = self.env_map.start_pose[0:3]
        return self.base_reset()

    def check_done(self):
        return self.base_check_done()

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


class DiscriminatingImgKernel:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.resolution = int(1/sim_conf.resolution)
        self.t_step = sim_conf.time_step
        self.velocity = 2 #TODO: make this a config param
        self.n_phi = 61  #TODO: add to conf file
        self.phi_range = np.pi #TODO: add to conf file
        self.half_block = 1 / (2*self.resolution)
        self.half_phi = self.phi_range / (2*self.n_phi)
        self.n_modes = 5 #TODO: add to conf file

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

    # config functions
    def build_qs(self):
        max_steer = 0.35
        ds = np.linspace(-max_steer, max_steer, self.n_modes)
        self.qs = self.velocity / 0.33 * np.tan(ds)

    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = kernel_loop(self.kernel, self.xs, self.ys, self.phis, self.n_modes, self.dynamics)

            # plt.figure(2)
            # plt.title(f"Kernel after loop: {z}")
            # phi_n = 30
            # img = self.kernel[:, :, phi_n].T - self.previous_kernel[:, :, phi_n].T
            # plt.imshow(img, origin='lower')
            # plt.pause(0.0001)

            # self.view_kernel(0, False)
        # self.save_kernel()

    def save_kernel(self, name="std_kernel"):
        np.save(f"SupervisorySafetySystem/Kernels/{name}.npy", self.kernel)
        print(f"Saved kernel to file")

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


# @njit(cache=True)
def build_dynamics_table(phis, qs, velocity, time, resolution):
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
        if safe:
            return False

    return True



class Kernel:
    def __init__(self, sim_conf):
        self.kernel = None
        self.resolution = int(1 / sim_conf.resolution)
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


@njit(cache=True)
def get_angles(n_beams=1000, fov=np.pi):
    return np.arange(n_beams) * fov / 999 -  np.ones(n_beams) * fov /2 

@njit(cache=True)
def get_trigs(n_beams, fov=np.pi):
    angles = np.arange(n_beams) * fov / 999 -  np.ones(n_beams) * fov /2 
    return np.sin(angles), np.cos(angles)

@njit(cache=True)
def convert_scan_xy(scan):
    sines, cosines = get_trigs(len(scan))
    xs = scan * sines
    ys = scan * cosines    
    return xs, ys

class SafetyPlannerPP:
    def __init__(self):
        self.d_max = 0.4 # radians  
        self.v = 2
        self.kernel = None

    def plan(self, obs):
        pp_action = self.run_pure_pursuit(obs['state'])
        # pp_action = self.take_random_action()
        state = np.array(obs['state'])

        safe, next_state = check_init_action(state, pp_action, self.kernel)
        if safe:
            # self.plot_single_flower(obs, next_state)
            return pp_action

        dw = self.generate_dw()
        valids = simulate_and_classify(state, dw, self.kernel)
        if not valids.any():
            print('No Valid options')
            print(f"State: {obs['state']}")
            return pp_action
        
        action = modify_action(pp_action, valids, dw)
        # print(f"Valids: {valids} -> new action: {action}")


        return action

    def take_random_action(self):
        np.random.seed()
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
    







if __name__ == "__main__":
    run_test_loop()
    # construct_kernel()
