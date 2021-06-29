
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

        scan = obs['full_scan']
        state = obs['state']

        d = state[4]
        dw_ds = build_dynamic_window(d, self.max_steer, self.max_d_dot, 0.1)

        valid_window, starts, ends, v_valids = check_dw_vo(scan, dw_ds, d)

        # x1, y1 = segment_lidar_scan(scan)
        x1, y1 = convert_scan_xy(scan)

        if not valid_window.any():
            print(f"Massive problem: no valid answers")
            return pp_action
        
        valid_dt = edt(valid_window)
        new_action = modify_action(pp_action, valid_window, dw_ds, valid_dt)

        return new_action

    def run_safety_check_plot(self, obs, pp_action):
        scan = obs['full_scan'] 
        state = obs['state']

        # np.save("SafeData/lidar_scan", scan)

        d = state[4]
        dw_ds = build_dynamic_window(d, self.max_steer, self.max_d_dot, 0.1)

        valid_window, obses, v_valids, pts = check_dw_distance_obs(scan, dw_ds, state)

        x1, y1 = convert_scan_xy(scan)

        if not valid_window.any():
            print(f"Massive problem: no valid answers")
            # self.plot_lidar_scan_vo(x1, y1, scan, starts, ends)
            self.plot_vd_window(v_valids, dw_ds, valid_window, pp_action, [0, 0], d)
            self.plot_flower(scan, obses, d, dw_ds, [0, 0])
            plt.show()
            return np.array([0, 0])  
                  
        valid_dt = edt(valid_window)
        new_action = modify_action(pp_action, valid_window, dw_ds, valid_dt)

        self.plot_flower(scan, obses, d, dw_ds, new_action, pts)
        # self.plot_vd_window(v_valids, dw_ds, valid_window, pp_action, new_action, d)

        if pp_action[0] != new_action[0]:
            plt.show()
        # plt.show()

        return new_action

    def plot_vd_window(self, v_valids, dw_ds, valid_window, pp_action, new_action, d0):
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
        
        xs = np.linspace(-np.pi/2, np.pi/2, 50)
        for j, x in enumerate(xs):
            x_p = np.sin(x)
            y_p = np.cos(x)
            if v_valids[j]:
                plt.plot(x_p, y_p, 'x', color='green', markersize=14)
            else:
                plt.plot(x_p, y_p, 'x', color='red', markersize=14)

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


        # ts = np.linspace(0, 0.1*3, 10)
        # dus = np.linspace(dw_ds[0], dw_ds[-1], 19)
        # for du in dus:
        #     xs, ys = run_model(d0, du, ts)
        #     plt.plot(xs, ys, '--')   
        for pt in pts:
            x_p = [0, pt[0]]
            y_p = [0, pt[1]]
            plt.plot(x_p, y_p, '--')

        # ts = np.linspace(0, 0.6, 10)
        # du = new_action[0]
        # xs, ys = run_model(d0, du, ts)
        # plt.plot(xs, ys, linewidth=2)         

        for obs in obses:
            obs.plot_obs_pts()

        plt.pause(0.0001)
        # plt.pause(0.1)



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
    max_window_size = 5
    window_sz = int(min(max_window_size, max(valid_dt)-1))
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

    n_delta_pts = 50 
    ds = np.linspace(ldb, udb, n_delta_pts)

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

def run_step(x, a):
    n_steps = 3
    for i in range(10*n_steps):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

    return x 


def generate_obses(scan):
    xs, ys = segment_lidar_scan(scan)
    scan_pts = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
    d_cone = 2 # size to consider an obstacle

    new_scan = (xs**2+ys**2)**0.5
    inds = np.arange(len(new_scan))

    obses = []
    for i in range(len(new_scan)-1):
        pt1 = scan_pts[i]
        pt2 = scan_pts[i+1]
        if new_scan[i] > d_cone and new_scan[i+1] > d_cone:
            continue
        x_lim = 0.5
        if pt1[0] < -x_lim and pt2[0] < -x_lim:
            continue
        if pt1[0] > x_lim and pt2[0] > x_lim:
            continue
        

        obs = Obstacle(pt1, pt2)
        obses.append(obs)

    return obses



def check_dw_distance_obs(scan, dw_ds, state):
    speed = max(state[3], 1)

    valid_ds = np.ones_like(dw_ds)

    pts = np.zeros((len(dw_ds), 2))
    x_state = np.array([0, 0, state[2], state[3], state[4]])
    for i, d in enumerate(dw_ds):
        x_prime = run_step(x_state, np.array([d, speed]))
        pts[i] = x_prime[0:2] 

    obses = generate_obses(scan)

    v_window = np.linspace(-np.pi/2, np.pi/2, 50)
    v_valids = np.ones_like(v_window)

    for i, pt in enumerate(pts):
        for obs in obses:
            safe = obs.check_location_safe(pt)
            valid_ds[i] = safe 
        
    return valid_ds, obses, v_valids, pts

class Obstacle:
    def __init__(self, start, end):
        buffer = 0.1 
        self.start_x = start[0] - buffer 
        self.start_y = start[1]
        self.end_x = end[0] + buffer
        self.end_y = end[1]
        
    def check_location_safe(self, pt):
        x, y = pt
        if x < self.start_x or x > self.end_x:
            # obstacle is not considered 
            return True 
        
        if y > self.start_y and y > self.end_y:
            # it will definitely crash, in line with obs
            return False 

        if x > 0:
            ret_val = self.check_right_side(pt) 
        elif x < 0:
            ret_val = self.check_left_side(pt)
        elif x == 0:
            ret_val = self.check_both_sides(pt)

        if ret_val is False:
            print("Unsafe action: pt")

        return ret_val
        
         

    def check_right_side(self, pt):
        # x > 0 and x < self.x_end 
        w = self.end_x - pt[0]
        distance_required = find_distance_obs(w)
        d_to_obs = self.end_y - pt[1] # self.end_y should be bigger
        if distance_required < d_to_obs:
            return True # it is safe 
        else:
            return False
            
            
    def check_left_side(self, pt):
        # x < 0 and x > self.start_x 
        w = pt[0] - self.start_x
        distance_required = find_distance_obs(w)
        d_to_obs = self.start_y - pt[1] # self.end_y should be bigger
        if distance_required < d_to_obs:
            return True # it is safe 
        else:
            return False
                        
    def check_both_sides(self, pt):
        # pt[0] == 0
        w = - self.start_x
        distance_left = find_distance_obs(w)
        d_min_left = self.start_y - pt[1] # self.end_y should be bigger
        if distance_left < d_min_left:
            return False 

        w = self.end_x
        distance_right = find_distance_obs(w)
        d_min_right = self.end_y - pt[1] # self.end_y should be bigger
        if distance_right < d_min_right:
            return False 
        
        return True
        

    def plot_obs_pts(self):
        if self.start_x > 0.8 or self.end_x < -0.8:
            return

        pt_left = [self.start_x, self.start_y]
        pt_right = [self.end_x, self.end_y]
        c_x = (self.start_x + self.end_x) / 2

        w_left = c_x - self.start_x 
        d_left = find_distance_obs(w_left)
        pt_left_center = [c_x-0.01, self.start_y-d_left]
        
        w_right = self.end_x - c_x 
        d_right = find_distance_obs(w_right)
        pt_right_center = [c_x+0.01,self.end_y-d_right]

        pts = np.vstack((pt_left,pt_right,pt_right_center, pt_left_center, pt_left))

        plt.plot(pts[:, 0], pts[:, 1])
        
             
        


def find_distance_obs(w, L=0.33, d_max=0.4):
    ld = np.sqrt(w*2*L/np.tan(d_max))
    distance = ((ld)**2 - (w**2))**0.5
    return distance


@njit(cache=True)
def check_action_safe(valid_window, d_idx, window=5):
    i_min = max(0, d_idx-window)
    i_max = min(len(valid_window)-1, d_idx+window)
    valids = valid_window[i_min:i_max]
    if valids.all():
        return True 
    return False


