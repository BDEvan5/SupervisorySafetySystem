
import numpy as np
from numba import njit, jit 
from numba.typed import List
from matplotlib import pyplot as plt
import csv
from scipy.ndimage import distance_transform_edt as edt

import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.speed_utils import calculate_speed, calculate_safe_speed
from toy_auto_race.Utils import pure_pursuit_utils

from toy_auto_race.lidar_viz import *




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

        self.vis = LidarViz(1000)
        self.old_steers = []
        self.new_steers = []

        self.last_scan = None
        self.new_action = None
        self.col_vals = None
        self.o_col_vals = None
        self.o_action = None

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

        # pp_action[1] = max(pp_action[1], state[3])
        action = self.run_safety_check(obs, pp_action)

        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        return action 

    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()
        self.step = 0

    def show_history(self, wait=False):
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

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

    def run_safety_check(self, obs, pp_action):

        scan = obs['scan']
        state = obs['state']

        v = state[3]
        d = state[4]
        dw_ds = build_dynamic_window(v, d, self.max_v, self.max_steer, self.max_a, self.max_d_dot, 0.1)

        valid_window, starts, ends = check_dw_vo(scan, dw_ds)

        # x1, y1 = segment_lidar_scan(scan)
        x1, y1 = convert_scan_xy(scan)

        new_action = modify_action(pp_action, valid_window, dw_ds)

        # self.plot_valid_window(dw_ds, valid_window, pp_action, new_action)

        # self.plot_lidar_scan_vo(x1, y1, scan, starts, ends)

        return new_action



    def plot_valid_window(self, dw_ds, valid_window, pp_action, new_action):
        plt.figure(1)
        plt.clf()
        plt.title("Valid window")

        sf = 1.1
        plt.xlim([-0.45, 0.45])
        plt.ylim([0, 2])

        for j, d in enumerate(dw_ds):
            if valid_window[j]:
                plt.plot(d, 1, 'x', color='green', markersize=14)
            else:
                plt.plot(d, 1, 'x', color='red', markersize=14)

        plt.plot(pp_action[0], 1, '+', color='red', markersize=22)
        plt.plot(new_action[0], 1, '*', color='green', markersize=16)

        # plt.show()
        plt.pause(0.0001)

    def plot_lidar_scan_vo(self, xs, ys, scan, starts, ends):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: {self.step}')

        plt.ylim([0, 8])
        # plt.xlim([-1.5, 1.5])
        plt.xlim([-4, 4])
        # plt.xlim([-1.5, 1.5])
        # plt.ylim([0, 3])
        plt.plot(xs, ys, '-+')

        sines, cosines = get_trigs(len(scan))
        for s, e in zip(starts, ends):
            xss = [0, scan[s]*sines[s], scan[e]*sines[e], 0]
            yss = [0, scan[s]*cosines[s], scan[e]*cosines[e], 0]
            plt.plot(xss, yss, '-+')

        plt.pause(0.0001)


# @njit(cache=True)
def segment_lidar_scan2(scan):
    """ 
    Takes a lidar scan and reduces it to a set of points that make straight lines 
    TODO: possibly change implmentation to work completely in r, ths 
    """
    xs, ys = convert_scan_xy(scan)
    #TODO: probably useful to be able to get the diffs straight from the r,t h representation
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
    i_pts = [0]
    d_thresh = 0.2

    inds = diffs > d_thresh
    all_inds = np.arange(999)
    i_pts = all_inds[inds]
    i_pts = np.append(i_pts, i_pts+np.ones_like(i_pts))
    i_pts = np.insert(i_pts, 0, 0)
    i_pts = np.insert(i_pts, -1, 999)

    i_pts = np.sort(i_pts)

    # for i in range(len(diffs)):
    #     if diffs[i] > d_thresh:
    #         i_pts.append(i)
    #         i_pts.append(i+1)
    # i_pts.append(len(scan)-1)

    # if len(i_pts) < 3:
    #     i_pts = [0]
    #     d_thresh = 0.1
    #     for i in range(len(diffs)):
    #         if diffs[i] > d_thresh:
    #             i_pts.append(i)
    #             i_pts.append(i+1)
    #     i_pts.append(len(scan)-1)
    
    # i_pts = np.array(i_pts)
    x_pts = xs[i_pts]
    y_pts = ys[i_pts]

    return x_pts, y_pts

# @njit(cache=True)
def segment_lidar_scan(scan):
    """ 
    Takes a lidar scan and reduces it to a set of points that make straight lines 
    TODO: possibly change implmentation to work completely in r, ths 
    """
    xs, ys = convert_scan_xy(scan)
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
    i_pts = [0]
    d_thresh = 0.3
    for i in range(len(diffs)):
        if diffs[i] > d_thresh:
            i_pts.append(i)
            i_pts.append(i+1)
    i_pts.append(len(scan)-1)

    if len(i_pts) < 3:
        i_pts.append(np.argmax(scan))
        

    i_pts = np.array(i_pts)
    x_pts = xs[i_pts]
    y_pts = ys[i_pts]

    return x_pts, y_pts


# @jit(cache=True, nopython=False)
def modify_action(pp_action, valid_window, dw_ds):
    d_idx = np.count_nonzero(dw_ds[dw_ds<pp_action[0]])
    if not valid_window.any():
        print(f"Massive problem: no valid answers")

        return pp_action
    if check_action_safe(valid_window, d_idx):
        return pp_action 
    else: 
        d_idx_search = np.argmin(np.abs(dw_ds))
        d_idx = find_new_action(valid_window, d_idx_search)
        new_action = np.array([dw_ds[d_idx], 3])
        return new_action


# @jit(cache=True, nopython=False)
# can't jit due to edt call
def find_new_action(valid_window, d_idx):
    d_size = len(valid_window)
    dt = edt(valid_window)
    window_sz = int(min(5, max(dt)-1))
    for i in range(len(valid_window)): # search d space
        p_d = min(d_size-1, d_idx+i)
        if check_action_safe(valid_window, p_d, window_sz):
            return p_d 
        n_d = max(0, d_idx-i)
        if check_action_safe(valid_window, n_d, window_sz):
            return n_d 
    print(f"No Action Found: redo Search")

@njit(cache=True) 
def build_dynamic_window(v, delta, max_v, max_steer, max_a, max_d_dot, dt):
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    n_delta_pts = 50 
    
    ds = np.linspace(ldb, udb, n_delta_pts)

    return ds

@njit(cache=True) 
def check_dw_vo(scan, dw_ds):
    d_cone = 1.6
    L = 0.33

    angles = np.arange(1000) * np.pi / 999 -  np.ones(1000) * np.pi/2 

    valid_ds = np.ones_like(dw_ds)
    inds = np.arange(1000)

    invalids = inds[scan<d_cone]
    starts1 = invalids[1:][invalids[1:] != invalids[:-1] + 1]
    starts = np.concatenate((np.zeros(1), starts1))
    ends1 = invalids[:-1][invalids[1:] != invalids[:-1] + 1]
    ends = np.append(ends1, invalids[-1])

    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)
        d_min = np.arctan(2*L*np.sin(angles[s])/scan[s])
        d_max = np.arctan(2*L*np.sin(angles[e])/scan[e])

        d_min = max(d_min, dw_ds[0])
        d_max = min(d_max, dw_ds[-1]+0.001)

        i_min = np.count_nonzero(dw_ds[dw_ds<d_min])
        i_max = np.count_nonzero(dw_ds[dw_ds<d_max])

        valid_ds[i_min:i_max] = False

    return valid_ds, starts, ends
        

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

@njit(cache=True)
def check_action_safe(valid_window, d_idx, window=5):
    i_min = max(0, d_idx-window)
    i_max = min(len(valid_window)-1, d_idx+window)
    valids = valid_window[i_min:i_max]
    if valids.all():
        return True 
    return False


