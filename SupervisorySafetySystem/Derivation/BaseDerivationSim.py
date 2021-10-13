import numpy as np
from matplotlib import pyplot as plt
from numba import njit, jit
import yaml
from argparse import Namespace

from SupervisorySafetySystem.Simulator.ForestSim import ForestMap 
from SupervisorySafetySystem.Simulator.LaserScanner import ScanSimulator
import LearningLocalPlanning.LibFunctions as lib


def load_conf(fname):
    full_path =  fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


class BaseSim:
    """Base class for simulation of safety system
    Variables to be added:
        state

    Methods to be defined:
        reset: reset the state and call base reset
        check_done: additional done checks like turning around
        update_state: implement the specific kinematic model
     """
    def __init__(self):
        self.env_map = ForestMap("forest2")
        sim_conf = load_conf("config/fgm_config")
        self.sim_conf = sim_conf

        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.init_sim_map(self.env_map)

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v

        self.done = False
        self.reward = 0
        self.done_reason = "Not set"

        self.pos_history = []


    def step(self, action):
        for _ in range(10):
            self.state = self.update_state(action, 0.008)

            if self.check_done():
                break

        self.pos_history.append(self.state[0:2])
        obs = self.get_observation()


        return obs, self.reward, self.done, None 

    def base_reset(self):
        self.pos_history.clear()
        self.pos_history.append(self.state[0:2])
        self.done_reason = "Not set"
        self.done = False

        self.env_map.add_obstacles()
        self.scan_sim.dt = self.env_map.set_dt()
         
        return self.get_observation()
   
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        observation = {}
        observation['state'] = self.state
        observation['scan'] = self.scan_sim.scan(self.state[0:3],10) 
        observation['full_scan'] = self.scan_sim.scan(self.state[0:3], 1000)
        observation['reward'] = self.reward
        obs_pts1, obs_pts2 =  self.env_map.get_relative_obs_pts(self.state[0:2])
        observation['obs_pts1'] = obs_pts1
        observation['obs_pts2'] = obs_pts2

        return observation

    def base_check_done(self):
        """
        Checks if the episode in the forest is complete 

        Returns:
            done (bool): a flag if the ep is done
        """
        self.reward = 0 # normal
        # check if finished lap
        dx = self.state[0] - self.env_map.start_pose[0]
        dx_lim = self.env_map.forest_width * 0.5
        if dx < dx_lim and self.state[1] > self.env_map.end_y:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"

        # check crash
        elif self.env_map.check_scan_location(self.state[0:2]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.state[0]:.2f}, {self.state[1]:.2f}]"

        # check steps
        # elif self.steps > self.max_steps:
        #     self.done = True
        #     self.reward = -1
        #     self.done_reason = f"Max steps"
        # check orientation
        # elif abs(self.state[2]) > 0.66*np.pi:
        #     self.done = True
        #     self.done_reason = f"Vehicle turned around"
        #     self.reward = -1

        # elif self.action[1] == 0 and self.state[3] < 1:
        #     self.done = True
        #     self.reward = -1
        #     self.done_reason = "Zero velocity"

        return self.done

    def render_ep(self, show=False):
        plt.figure(1)
        plt.clf()

        self.env_map.render_map(1)
        plt.title(f"Racing episode: {self.done_reason}")

        xs, ys = scale_to_plot(np.array(self.pos_history))
        plt.plot(xs, ys, 'x-')


        if show:
            plt.show()

    def render_pose(self, show=False):
        plt.figure(1)
        # plt.clf()

        self.env_map.render_map(1)

        xs, ys = scale_to_plot(np.array([self.state[0:2]]))
        plt.plot(xs, ys, 'x', markersize=16)
        plt.arrow(xs[0], ys[0], np.sin(self.state[2])*20, np.cos(self.state[2])*20, head_width=0.05, head_length=0.1, fc='k', ec='k')

        plt.pause(0.0001)

def scale_to_plot(pts):
    resolution = 0.05 
    xs = pts[:, 0] / resolution
    ys = pts[:, 1] / resolution
    return xs, ys 

@njit(cache=True)
def calculate_progress(point, wpts, diffs, l2s, ss):
    dots = np.empty((wpts.shape[0]-1, ))
    dots_shape = dots.shape[0]
    for i in range(dots_shape):
        dots[i] = np.dot((point - wpts[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0  #np.clip, unsupported
    
    projections = wpts[:-1,:] + (t*diffs.T).T

    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))

    min_dist_segment = np.argmin(dists)
    dist_from_cur_pt = dists[min_dist_segment]

    s = ss[min_dist_segment] + dist_from_cur_pt
    # print(F"{min_dist_segment} --> SS: {ss[min_dist_segment]}, curr_pt: {dist_from_cur_pt}")

    s = s / ss[-1]

    return s 

