
import numpy as np
from numba import njit, jit
from matplotlib import pyplot as plt
import csv

from numpy.core.numerictypes import maximum_sctype

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

        # check if current corridor is safe
        scan = obs['scan']
        # scan *= 0.95
        state = obs['state']


        v = state[3]
        d = state[4]
        dw_ds = build_dynamic_window(v, d, self.max_v, self.max_steer, self.max_a, self.max_d_dot, 0.1)


        rs, ths  = segment_lidar_scan(scan)
        x1, y1 = convert_polar_xy(rs, ths)
        rs, ths  = create_safety_cones(rs, ths )

        # rs, ths = clean_r_list(rs, ths)
        rs, ths = convexification(rs, ths)

        valid_window, end_pts = check_dw_clean(dw_ds, rs, ths, d)

        new_action = self.modify_action(pp_action, valid_window, dw_ds)

        self.plot_valid_window(dw_ds, valid_window, pp_action, new_action)

        self.plot_lidar_scan_clean(x1, y1, end_pts, rs, ths)
        # self.plot_lidar_scan_clean(x_pts, y_pts, end_pts, rs, ths)

        plt.show()
        # if not valid_window.any():
        #     plt.show()

        # if new_action[0] != pp_action[0]:
        #     plt.show()

        return new_action

    def modify_action(self, pp_action, valid_window, dw_ds):
        d_idx = action_to_ind(pp_action, dw_ds)
        if not valid_window.any():
            print(f"Massive problem: no valid answers")

            return pp_action
        if check_action_safe(valid_window, d_idx):
            return pp_action 
        else: 
            d_idx_search = np.argmin(np.abs(dw_ds))
            d_idx = self.find_new_action(valid_window, d_idx_search)
            new_action = np.array([dw_ds[d_idx], 3])
            return new_action

    def find_new_action(self, valid_window, d_idx):
        d_size = len(valid_window)
        for i in range(len(valid_window)): # search d space
            p_d = min(d_size-1, d_idx+i)
            if check_action_safe(valid_window, p_d):
                return p_d 
            n_d = max(0, d_idx-i)
            if check_action_safe(valid_window, n_d):
                return n_d 



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

    def plot_lidar_scan_clean(self, xs, ys, end_pts, rs, ths):
        plt.figure(2)
        plt.clf()
        plt.title(f'Lidar Scan: {self.step}')

        plt.ylim([0, 10])
        plt.xlim([-1.5, 1.5])
        # plt.xlim([-1.5, 1.5])
        # plt.ylim([0, 3])
        plt.plot(xs, ys, '-+')

        xs, ys = convert_polar_xy(rs, ths)
        plt.plot(xs, ys, '-+', linewidth=2)

        xs = end_pts[:, 0].flatten()
        ys = end_pts[:, 1].flatten()
        for x, y in zip(xs, ys):
            x_p = [0, x]
            y_p = [0, y]
            plt.plot(x_p, y_p, '--')

        plt.pause(0.0001)
        # plt.show()



def convexification(rs, ths):
    xs, ys = convert_polar_xy(rs, ths)
    ms, cs = convert_xy_mc(xs, ys)
    o_ths = np.copy(ths)

    idx1 = np.count_nonzero(xs[xs<0]) -1
    idx2 = idx1 +1

    N = len(ths)

    cur_angle = ths[idx1]
    for i in range(idx1-1, -1, -1):
        cur_angle = min(cur_angle-0.0001, ths[i])
        ths[i] = cur_angle
        if ths[i] != o_ths[i]:
            r_pt = calculate_intersection(ms[i-1], cs[i-1], ths[i])
            r = np.sqrt(np.sum(np.power(r_pt, 2)))
            rs[i] = r

    cur_angle = ths[idx2]
    for i in range(idx2+1, N):
        cur_angle = max(cur_angle+0.0001, ths[i])
        ths[i] = cur_angle
        if ths[i] != o_ths[i] and i != N-1:
            r_pt = calculate_intersection(ms[i], cs[i], ths[i])
            r = np.sqrt(np.sum(np.power(r_pt, 2)))
            rs[i] = r
    
    return rs, ths

# def find_new_pt(ms, cs, new_th, i):
    # r_pt = calculate_intersection(ms[i], cs[i], new_th)
    # r = np.sqrt(np.sum(np.power(r_pt, 2)))


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

    rs, ths = convert_xys_rths(x_pts, y_pts)

    return rs, ths 

# @njit(cache=True) 
def create_safety_cones_old(rs, ths):
    x_pts, y_pts = convert_polar_xy(rs, ths)
    N = len(x_pts)
    L = 0.33
    max_steer = 0.4

    new_x = x_pts
    new_y = y_pts

    y_thresh = 0.1
    n_new_pts = 0
    x_buff = 0
    x_n = min(x_pts)
    x_p = max(x_pts)
    for i in range(N-1):
        if abs(y_pts[i] - y_pts[i+1]) < y_thresh:
            w = (x_pts[i+1] - x_pts[i])/2 
            l_d = np.sqrt(w * 2 * L / np.tan(max_steer))
            d = np.sqrt(l_d **2 - w**2)

            x = x_pts[i] + w
            y = (y_pts[i] + y_pts[i+1])/2 - d * 2
            r, th = convert_xys_rths(x, y)

            # changes the x, y vals accoridng to buffer pt
            new_x[i+n_new_pts] = max(new_x[i+n_new_pts] - x_buff, x_n)
            new_x[i+n_new_pts+1] = min(new_x[i+n_new_pts+1]+x_buff, x_p) 

            # inserts a new point into the list 

            new_x = np.insert(new_x, i+1+n_new_pts, x)
            new_y = np.insert(new_y, i+1+n_new_pts, y)
            n_new_pts += 1

    return new_x, new_y

# @njit(cache=True) 
def create_safety_cones(rs, ths):
    x_pts, y_pts = convert_polar_xy(rs, ths)
    ms, cs = convert_xy_mc(x_pts, y_pts)
    N = len(x_pts)
    L = 0.33
    max_steer = 0.4

    y_thresh = 0.1
    n_new_pts = 0
    x_buff = 0
    x_n = min(x_pts)
    x_p = max(x_pts)
    for i in range(N-1):
        if abs(y_pts[i] - y_pts[i+1]) < y_thresh:
            w = (x_pts[i+1] - x_pts[i])/2 
            l_d = np.sqrt(w * 2 * L / np.tan(max_steer))
            d = np.sqrt(l_d **2 - w**2)

            x = x_pts[i] + w
            y = (y_pts[i] + y_pts[i+1])/2 - d * 2
            r, th = convert_xys_rths(x, y)
            if y < 0:  # keep track of angle
                th += np.pi

            # changes the x, y vals accoridng to buffer pt
            x1, x2 = x_pts[i:i+2]
            y1, y2 = y_pts[i:i+2]
            x1 = max(x1 - x_buff, x_n)
            x2 = min(x2+x_buff, x_p) 

            # move left point 
            r1, th1 = convert_xys_rths(x1, y1)
            rs[i+n_new_pts] = r1 
            ths[i+n_new_pts] = th1 

            # adjust left point in case it becomes unconvex
            ths[i+n_new_pts-1] = min(th1, ths[i+n_new_pts-1])
            r_pt = calculate_intersection(ms[i], cs[i], ths[i+n_new_pts-1])
            rs[i+n_new_pts-1] = np.sqrt(np.sum(np.power(r_pt, 2)))

            # move right point 
            r2, th2 = convert_xys_rths(x2, y2)
            rs[i+n_new_pts+1] = r2  
            ths[i+n_new_pts+1] = th2

            # adjust right point in case of unconvex
            if i < len(ms) - 2:
                ths[i+n_new_pts+2] = max(th1, ths[i+n_new_pts+2])
                r_pt = calculate_intersection(ms[i+2], cs[i+2], ths[i+n_new_pts+2])
                rs[i+n_new_pts+2] = np.sqrt(np.sum(np.power(r_pt, 2)))

            # inserts a new point into the list 
            rs = np.insert(rs, i+1+n_new_pts, r)
            ths = np.insert(ths, i+1+n_new_pts, th)

            if th < 0: 
                # look at previous angles
                ths[i+n_new_pts] = min(ths[i+n_new_pts], th)
            else:
                # check the next pt
                ths[i+n_new_pts+2] = max(ths[i+n_new_pts+2], th)

            n_new_pts += 1

    return rs, ths


@njit(cache=True) 
def build_dynamic_window(v, delta, max_v, max_steer, max_a, max_d_dot, dt):
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    n_delta_pts = 50 
    
    ds = np.linspace(ldb, udb, n_delta_pts)

    return ds


# @jit(cache=True)
def check_dw_clean(dw_ds, rs, ths, o_d):
    dt = 0.1
    n_steps = 2
    valids = np.empty( len(dw_ds))
    end_pts = np.empty((len(dw_ds), 2))
    xs, ys = convert_polar_xy(rs, ths) 
    ms, cs = convert_xy_mc(xs, ys)
    for j, d in enumerate(dw_ds):
        t_xs, t_ys = predict_trajectory(d, n_steps, dt, o_d)
        safe, pt = check_pt_safe_clean(t_xs[-1], t_ys[-1], ths, ms, cs)

        valids[j] = safe 
        # end_pts[j, 0] = t_xs[-1]
        # end_pts[j, 1] = t_ys[-1]

        # add the intersection points
        end_pts[j, 0] = pt[0]
        end_pts[j, 1] = pt[1]

    return valids, end_pts

@njit(cache=True)
def predict_trajectory(d, n_steps, dt, o_d, v=3):
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    speed = 3
    x = np.array([0, 0, 0, speed, o_d])
    ref = np.array([d, speed])
    for i in range(0, n_steps):
        for j in range(10):
            u = control_system(x, ref)
            x = update_kinematic_state(x, u, dt/10)
        xs[i] = x[0]
        ys[i] = x[1]
    
    return xs, ys



@njit(cache=True)
def convert_xys_rths(xs, ys):
    rs = np.sqrt(np.power(xs, 2) + np.power(ys, 2))
    ths = np.arctan(xs/ys)

    return rs, ths

@njit(cache=True)
def convert_polar_xy(rs, ths):
    xs = rs * np.sin(ths)
    ys = rs * np.cos(ths)

    return xs, ys

@njit(cache=True)
def convert_xy_mc(xs, ys):
    ms = (ys[1:] - ys[:-1])/(xs[1:] - xs[:-1])
    cs = ys[:-1] - ms * xs[:-1]

    return ms, cs 


@njit(cache=True)
def calculate_intersection(m, c, th):
    m_r = np.tan(np.pi/2 - th)
    x = c / (m_r - m)
    y = m_r * x 

    return np.array([x, y])

# @njit(cache=True)
def check_pt_safe_clean(x, y, ths, ms, cs):
    angle = np.arctan(x/y) # range -90, 90
    dths = ths - angle*np.ones_like(ths)
    idx = np.argmin(np.abs(dths))
    if angle < ths[idx]:
        # in this case, it is the previous segment
        idx -= 1 

    m = ms[idx]
    c = cs[idx]

    r = (x**2 + y**2)**0.5

    pt = calculate_intersection(m, c, angle)
    r_intersection = np.sqrt(np.sum(np.power(pt, 2)))

    if r > r_intersection:
        return False, pt  
    return True, pt


@njit(cache=True)
def get_trigs(n_beams, fov=np.pi):
    angles = np.empty(n_beams)
    for i in range(n_beams):
        angles[i] = -fov/2 + fov/(n_beams-1) * i
    sines = np.sin(angles)
    cosines = np.cos(angles)

    return sines, cosines

@njit(cache=True)
def convert_scan_xy(scan):
    sines, cosines = get_trigs(len(scan))
    xs = scan * sines
    ys = scan * cosines    
    return xs, ys

@njit(cache=True)
def check_action_safe(valid_window, d_idx):
    window = 5
    valids = valid_window[d_idx-window:d_idx+window]
    if valids.all():
        return True 
    return False

@njit(cache=True)
def action_to_ind(action, dw_ds):
    d_idx = np.count_nonzero(dw_ds[dw_ds<action[0]])
    return d_idx

@njit(cache=True)
def update_kinematic_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
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

@njit(cache=True)
def control_system(state, action, max_v=7, max_steer=0.4, max_a=8.5, max_d_dot=3.2):
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
    

