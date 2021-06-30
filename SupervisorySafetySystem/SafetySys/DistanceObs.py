
import numpy as np
from numba import njit, jit 
from matplotlib import pyplot as plt
import csv
from scipy.ndimage import distance_transform_edt as edt

import SupervisorySafetySystem.LibFunctions as lib
from toy_auto_race.Utils import pure_pursuit_utils

from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state
from SupervisorySafetySystem.SafetySys.safety_utils import *
from SupervisorySafetySystem.SafetySys.LidarProcessing import segment_lidar_scan
from SupervisorySafetySystem.SafetySys.Obstacle import Obstacle, OrientationObstacle



class SafetyPP:
    def __init__(self, sim_conf) -> None:
        self.name = "Safety Car"
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.max_steer = sim_conf.max_steer
        self.max_v = sim_conf.max_v
        self.max_d_dot = sim_conf.max_d_dot

        self.v_gain = 0.5
        self.lookahead = 1.6
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self.aim_pts = []

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, 2])
        else:
            return None

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        self.aim_pts.append(lookahead_point[0:2])

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)


        speed = 3
        # speed = calculate_speed(steering_angle)

        return np.array([steering_angle, speed])

    def reset_lap(self):
        self.aim_pts.clear()

    def plan(self, env_map):
        if self.waypoints is None:
            track = []
            filename = 'maps/' + env_map.map_name + "_opti.csv"
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                for lines in csvFile:  
                    track.append(lines)

            track = np.array(track)
            print(f"Track Loaded: {filename}")

            wpts = track[:, 1:3]
            vs = track[:, 5]

            self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)
            self.expand_wpts()

            return self.waypoints[:, 0:2]

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints[:, 0:2]
        # o_ss = self.ss
        o_vs = self.waypoints[:, 2]
        new_line = []
        # new_ss = []
        new_vs = []
        for i in range(len(self.waypoints)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                # ds = o_ss[i+1] - o_ss[i]
                # new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        wpts = np.array(new_line)
        # self.ss = np.array(new_ss)
        vs = np.array(new_vs)
        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)


class SafetyCar(SafetyPP):
    def __init__(self, sim_conf):
        SafetyPP.__init__(self, sim_conf)
        self.sim_conf = sim_conf # kept for optimisation
        self.n_beams = 1000
        self.step = 0

        safety_f = 0.9
        self.max_a = sim_conf.max_a * safety_f
        self.max_steer = sim_conf.max_steer

        self.old_steers = []
        self.new_steers = []

        self.fov = np.pi
        self.dth = self.fov / (self.n_beams-1)
        self.center_idx = int(self.n_beams/2)

        self.angles = np.empty(self.n_beams)
        for i in range(self.n_beams):
            self.angles[i] =  self.fov/(self.n_beams-1) * i

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)
        self.step += 1

        # action = self.run_safety_check(obs, pp_action)
        action = self.run_safety_check_plot(obs, pp_action)
        # action = run_safety_check(obs, pp_action, self.max_steer, self.max_d_dot)

        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        return action 

    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()
        self.step = 0

    def show_history(self, wait=False):

        plt.figure(5)
        plt.clf()
        plt.plot(self.old_steers)
        plt.plot(self.new_steers)
        plt.legend(['Old', 'New'])
        plt.title('Old and New Steering')
        plt.ylim([-0.5, 0.5])

        plt.pause(0.0001)
        if wait:
            plt.show()
        
    def show_lidar(self):
        pass

    def run_safety_check_plot(self, obs, pp_action):
        scan = obs['full_scan'] 
        state = obs['state']

        d = state[4]
        dw_ds = build_dynamic_window(d, self.max_steer, self.max_d_dot, 0.1)

        valid_window, obses, pts = check_dw_distance_obs(scan, dw_ds, state)

        if not valid_window.any():
            print(f"Massive problem: no valid answers")
            self.plot_vd_window(dw_ds, valid_window, pp_action, [0, 0], d)
            self.plot_flower(scan, obses, d, dw_ds, [0, 0], pts)
            plt.show()
            return np.array([0, 0])  
                  
        valid_dt = edt(valid_window)
        new_action = modify_action(pp_action, valid_window, dw_ds, valid_dt)

        self.plot_flower(scan, obses, d, dw_ds, new_action, pts)
        self.plot_vd_window(dw_ds, valid_window, pp_action, new_action, d)

        # if pp_action[0] != new_action[0]:
        #     plt.show()
        # # plt.show()
        if len(obses) > 0:
            plt.show()

        return new_action

    def plot_vd_window(self, dw_ds, valid_window, pp_action, new_action, d0):
        plt.figure(1)
        plt.clf()
        plt.title("Valid windows")

        plt.ylim([0, 2])

        for j, d in enumerate(dw_ds):
            if valid_window[j]:
                plt.plot(d, 1.5, 'x', color='green', markersize=14)
            else:
                plt.plot(d, 1.5, 'x', color='red', markersize=14)

        plt.plot(pp_action[0], 1.3, '+', color='red', markersize=22)
        plt.plot(new_action[0], 1.3, '*', color='green', markersize=16)
        plt.plot(d0, 1.7, '*', color='green', markersize=16)
        
        # xs = np.linspace(-np.pi/2, np.pi/2, 50)
        # for j, x in enumerate(xs):
        #     x_p = np.sin(x)
        #     y_p = np.cos(x)
        #     if v_valids[j]:
        #         plt.plot(x_p, y_p, 'x', color='green', markersize=14)
        #     else:
        #         plt.plot(x_p, y_p, 'x', color='red', markersize=14)

        t = 0.3
        x0, y0, _ = steering_model_clean(d0, pp_action[0], t)
        v_vec_x = [0, x0]
        v_vec_y = [0, y0]
        plt.plot(v_vec_x, v_vec_y, linewidth=4)

        x1, y1, _ = steering_model_clean(d0, new_action[0], t)

        v_vec_x = [x0, x1]
        v_vec_y = [y0, y1]
        plt.plot(v_vec_x, v_vec_y, linewidth=4)

        ts = np.linspace(0, 0.3, 10)
        dus = np.linspace(dw_ds[0], dw_ds[-1], 11)
        for du in dus:
            xs, ys = run_model(d0, du, ts)
            plt.plot(xs, ys, '--')      

        plt.text(-0.9, 1.8, f"d0: {d0:.4f}")
        plt.text(0, 1.8, f"du: {new_action[0]:.4f}")

        # plt.show()
        plt.pause(0.0001)

    def plot_flower(self, scan, obses, d0, dw_ds, new_action, pts):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: {self.step}')

        plt.ylim([0, 3])
        plt.xlim([-1.5, 1.5])
        xs, ys = convert_scan_xy(scan)
        plt.plot(xs, ys, '-+')

        x_seg, y_seg = segment_lidar_scan(scan)
        plt.plot(x_seg, y_seg, 'x', markersize=16)
        
        length = 0.2
        for pt in pts:
            x_p = [0, pt[0]]
            y_p = [0, pt[1]]
            th = pt[2]
            plt.arrow(pt[0], pt[1], np.sin(th)*length, np.cos(th)*length, head_width=0.03) 
            plt.plot(x_p, y_p, '--')

        plt.pause(0.0001)


@jit(cache=True)
def modify_action(pp_action, valid_window, dw_ds, valid_dt):
    d_idx = np.count_nonzero(dw_ds[dw_ds<pp_action[0]])
    if check_action_safe(valid_window, d_idx):
        new_action = pp_action 
    else: 
        d_idx_search = np.argmin(np.abs(dw_ds))
        d_idx = int(find_new_action(valid_window, d_idx_search, valid_dt))
        new_action = np.array([dw_ds[d_idx], 3])
        new_action = new_action
    return new_action

@jit(cache=True)
def find_new_action(valid_window, d_idx, valid_dt):
    d_size = len(valid_window)
    # max_window_size = 5
    # window_sz = int(min(max_window_size, max(valid_dt)-1))
    window_sz = 1
    for i in range(len(valid_window)): # search d space
        p_d = min(d_size-1, d_idx+i)
        if check_action_safe(valid_window, p_d, window_sz):
            return p_d 
        n_d = max(0, d_idx-i)
        if check_action_safe(valid_window, n_d, window_sz):
            return n_d 
    print(f"No Action Found: redo Search")

@njit(cache=True) 
def build_dynamic_window(delta, max_steer, max_d_dot, dt):
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    n_delta_pts = 5
    ds = np.linspace(ldb, udb, n_delta_pts)
    print(f"Dynamic Window built")

    return ds

def run_model(d0, du, ts):
    xs, ys = np.ones_like(ts), np.ones_like(ts)
    for i, t in enumerate(ts):
        xs[i], ys[i], _ = steering_model_clean(d0, du, t)

    return xs, ys

@njit(cache=True) 
def steering_model_clean(d0, du, t):
    speed = 3
    L = 0.33

    if du == d0:
        t_transient = 0 
        sign = 0
    else:
        t_transient = (du-d0)/3.2
        sign = t_transient / abs(t_transient)
        t_transient = abs(t_transient)

    ld_trans = speed * min(t, t_transient)
    d_follow = (3*d0 + 3.2*min(t, t_transient) * sign) / 3
    alpha_trans = np.arcsin(np.tan(d_follow)*ld_trans/(2*L))

    ld_prime = speed * max(t-t_transient, 0)
    alpha_prime = np.arcsin(np.tan(du)*ld_prime/(2*L))
    # reason for adding them is the old alpha is the base theta for the next step
    alpha_ss = alpha_trans + alpha_prime 

    x = ld_trans * np.sin(alpha_trans) + ld_prime*np.sin(alpha_ss)
    y = ld_trans * np.cos(alpha_trans) + ld_prime*np.cos(alpha_ss)

    th = speed / L * np.tan(du) * max(t-t_transient, 0) + speed / L * np.tan(d_follow) * t_transient 

    return x, y, th

def run_step(x, a, n_steps = 2):
    for i in range(10*n_steps):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 

def generate_obses(scan):
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

        obs = OrientationObstacle(pt1, pt2)
        obses.append(obs)

    return obses

def check_dw_distance_obs(scan, dw_ds, state):
    speed = max(state[3], 1)
    valid_ds = np.ones_like(dw_ds)

    pts = np.zeros((len(dw_ds), 3))
    x_state = np.array([0, 0, state[2], state[3], state[4]])
    for i, d in enumerate(dw_ds):
        x_prime = run_step(x_state, np.array([d, speed]), 1)
        pts[i] = x_prime[0:3] 

    obses = generate_obses(scan)

    plt.figure(3)
    plt.clf()
    for i, pt in enumerate(pts):
        safe = True 
        for obs in obses:
            # d_min, d_max = get_d_lims(dw_ds[i])
            # d_min, d_max = -0.4, 0.4
            d_min, d_max = dw_ds[0], dw_ds[-1]
            obs.run_check(pt[0:2], pt[2], d_min, d_max)
            obs.draw_obstacle()
            if not obs.is_safe():
                safe = False

            # obs.draw_situation(pt[0:2], pt[2], d_min, d_max)
            # if not obs.check_location_safe(pt[0:2], pt[2], d_min, d_max):
            #     safe = False
            # only set the valid window value once
        valid_ds[i] = safe 
        
    return valid_ds, obses, pts
  
        
def get_d_lims(d, t=0.1):
    sv = 3.2
    d_min = max(d-sv*t, -0.4)
    d_max = min(d+sv*t, 0.4)
    return d_min, d_max
    

@njit(cache=True)
def check_action_safe(valid_window, d_idx, window=5):
    i_min = max(0, d_idx-window)
    i_max = min(len(valid_window)-1, d_idx+window)
    valids = valid_window[i_min:i_max]
    if valids.all():
        return True 
    return False


