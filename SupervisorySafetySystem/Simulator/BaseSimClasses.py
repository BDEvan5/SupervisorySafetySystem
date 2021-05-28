import numpy as np 
from matplotlib import pyplot as plt
import os
from numba import njit

import LearningLocalPlanning.LibFunctions as lib
from LearningLocalPlanning.Simulator.LaserScanner import ScanSimulator


class CarModel:
    """
    A simple class which holds the state of a car and can update the dynamics based on the bicycle model

    Data Members:
        x: x location of vehicle on map
        y: y location of vehicle on map
        theta: orientation of vehicle
        velocity: 
        steering: delta steering angle
        th_dot: the change in orientation due to steering

    """
    def __init__(self, sim_conf):
        """
        Init function

        Args:
            sim_conf: a config namespace with relevant car parameters
        """
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0
        self.th_dot = 0

        self.prev_loc = 0

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, a, d_dot, dt):
        """
        Updates the internal state of the vehicle according to the kinematic equations for a bicycle model

        Args:
            a: acceleration
            d_dot: rate of change of steering angle
            dt: timestep in seconds

        """
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.th_dot = theta_dot
        dth = theta_dot * dt
        self.theta = lib.add_angles_complex(self.theta, dth)

        a = np.clip(a, -self.max_a, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

    def get_car_state(self):
        """
        Returns the state of the vehicle as an array

        Returns:
            state: [x, y, theta, velocity, steering]

        """
        state = []
        state.append(self.x) #0
        state.append(self.y)
        state.append(self.theta) # 2
        state.append(self.velocity) #3
        state.append(self.steering)  #4

        state = np.array(state)

        return state

    def reset_state(self, start_pose):
        """
        Resets the state of the vehicle

        Args:
            start_pose: the starting, [x, y, theta] to reset to
        """
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]
        self.velocity = 0
        self.steering = 0
        self.prev_loc = [self.x, self.y]



#TODO: move this to another location
class SimHistory:
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

    def show_forces(self):
        mu = self.sim_conf['car']['mu']
        m = self.sim_conf['car']['m']
        g = self.sim_conf['car']['g']
        l_f = self.sim_conf['car']['l_f']
        l_r = self.sim_conf['car']['l_r']
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.sim_conf['sim']['timestep'] * self.sim_conf['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)


class BaseSim:
    """
    Base simulator class

    Important parameters:
        timestep: how long the simulation steps for
        max_steps: the maximum amount of steps the sim can take

    Data members:
        car: a model of a car with the ability to update the dynamics
        scan_sim: a simulator for a laser scanner
        action: the current action which has been given
        history: a data logger for the history
    """
    def __init__(self, env_map, done_fcn, sim_conf):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.done_fcn = done_fcn
        self.env_map = env_map
        self.sim_conf = sim_conf #TODO: don't store the conf file, just use and throw away.
        self.n_obs = self.env_map.n_obs

        self.timestep = self.sim_conf.time_step
        self.max_steps = self.sim_conf.max_steps
        self.plan_steps = self.sim_conf.plan_steps

        self.state = np.zeros(5)
        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.init_sim_map(env_map)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.history = SimHistory(self.sim_conf)
        self.done_reason = ""

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v

    def step_plan(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """
        action = np.array(action)
        self.action = action
        for _ in range(self.plan_steps):
            u = control_system(self.state, action, self.max_v, self.max_steer, 8, 3.2)
            self.state = update_kinematic_state(self.state, u, self.timestep, self.wheelbase, self.max_steer, self.max_v)
            self.steps += 1 

            if self.done_fcn():
                break

        self.record_history(action)

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def record_history(self, action):
        self.action = action
        self.history.velocities.append(self.state[3])
        self.history.steering.append(self.state[4])
        self.history.positions.append(self.state[0:2])
        self.history.thetas.append(self.state[2])

    def reset(self, add_obs=True):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0
        self.reward = 0

        self.state[0:3] = self.env_map.start_pose 
        self.state[3:5] = np.zeros(2)

        self.history.reset_history()

        if add_obs:
            self.env_map.add_obstacles()

        # update the dt img in the scan simulator after obstacles have been added
        dt = self.env_map.set_dt()
        self.scan_sim.dt = dt

        return self.get_observation()

    def render(self, wait=False, name="No vehicle name set"):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)
        plt.title(name)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column(self.state[0:2])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.state[0]:.2f}, {self.state[1]:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.state[3]:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.state[2] * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.state[4]*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def get_target_obs(self):
        target = self.env_map.end_goal
        pos = self.state[0:2]
        base_angle = lib.get_bearing(pos, target) 
        angle = lib.sub_angles_complex(base_angle, self.state[2])

        em = self.env_map
        s = calculate_progress(pos, em.ref_pts, em.diffs, em.l2s, em.ss_normal)

        return [angle, s]
    
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        car_obs = self.state
        pose = car_obs[0:3]
        scan = self.scan_sim.scan(pose)
        target = self.get_target_obs()

        observation = {}
        observation['state'] = car_obs
        observation['scan'] = scan 
        observation['target'] = target
        observation['reward'] = self.reward

        # observation = np.concatenate([car_obs, target, scan, [self.reward]])
        return observation


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


#Dynamics functions
# @njit(cache=True)
def update_kinematic_state(x, u, dt, whlb, max_steer, max_v):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    dx = np.array([x[3]*np.sin(x[2]), # x
                x[3]*np.cos(x[2]), # y
                x[3]/whlb * np.tan(x[4]), # theta
                u[1], # velocity
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state

# @njit(cache=True)
def control_system(state, action, max_v, max_steer, max_a, max_d_dot):
    """
    Generates acceleration and steering velocity commands to follow a reference
    Note: the controller gains are hand tuned in the fcn

    Args:
        v_ref: the reference velocity to be followed
        d_ref: reference steering to be followed

    Returns:
        a: acceleration
        d_dot: the change in delta = steering velocity
    """
    # clip action
    v_ref = min(action[1], max_v)
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)

    kp_a = 10
    a = (v_ref-state[3])*kp_a
    
    kp_delta = 40
    d_dot = (d_ref-state[4])*kp_delta

    # clip actions
    a = min(a, max_a)
    a = max(a, -max_a)
    d_dot = min(d_dot, max_d_dot)
    d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot, a])

    return u



