from matplotlib import pyplot as plt 
import yaml, csv, os  
import numpy as np 
from numba import njit
from scipy import ndimage
from argparse import Namespace


from SupervisorySafetySystem.Simulator.BaseSimClasses import BaseSim





class ForestMap:
    def __init__(self, sim_conf) -> None:
        self.map_name = sim_conf.map_name 

        # map info
        self.resolution = 1/sim_conf.n_dx
        self.n_obs = sim_conf.n_obs
        self.forest_length = sim_conf.forest_length
        self.forest_width =  sim_conf.forest_width
        self.start_pose = np.array(sim_conf.start_pose)
        self.obs_size = sim_conf.obs_size
        self.obstacle_buffer = sim_conf.obstacle_buffer
        self.end_y = sim_conf.end_y

        self.map_height = int(self.forest_length / self.resolution)
        self.map_width = int(self.forest_width / self.resolution)
        self.end_goal = np.array([self.start_pose[0], self.end_y])
        self.map_img = np.zeros((self.map_width, self.map_height))
        self.dt_img = None
        self.set_dt()

        self.origin = [0, 0, 0] # for ScanSimulator

        self.ref_pts = None # std wpts that aren't expanded
        self.ss_normal = None # not expanded
        self.diffs = None
        self.l2s = None

        self.obs_pts = None

        self.load_center_pts()

    def add_obstacles(self):
        self.map_img = np.zeros((self.map_width, self.map_height))

        y_length = (self.end_y - self.obstacle_buffer*2 - self.start_pose[1] - self.obs_size)
        box_factor = 1.4
        y_box = y_length / (self.n_obs * box_factor)
        rands = np.random.random((self.n_obs, 2))
        # xs = rands[:, 0] * (self.forest_width-self.obs_size) 
        xs = rands[:, 0] * (self.forest_width-self.obs_size*2) + self.obs_size /2
        ys = rands[:, 1] * y_box
        y_start = self.start_pose[1] + self.obstacle_buffer
        y_pts = [y_start + y_box * box_factor * i for i in range(self.n_obs)]
        ys = ys + y_pts

        obs_locations = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        obs_size_px = int(self.obs_size/self.resolution)
        for location in obs_locations:
            x, y = self.xy_to_row_column(location)
            # print(f"Obstacle: ({location}): {x}, {y}")
            self.map_img[x:x+obs_size_px, y:y+obs_size_px] = 1

        # creates a list of obs pts that can be passed to the obs avoidance devel.    
        obs_pts2 = np.copy(obs_locations) 
        obs_pts2[:, 0] += np.ones_like(obs_pts2[:, 0]) * self.obs_size
        self.obs_pts = np.hstack((obs_locations, obs_pts2))
        
    def get_relative_obs_pts(self, pose):
        pts1 = self.obs_pts[:, 0:2] - pose[0:2]
        pts2 = self.obs_pts[:, 2:4] - pose[0:2]
        #TODO: add something to take vehicle orientation into account

        r_p1, r_p2 = [], []
        y_thresh = 3 # distance in front of vehicle to consider obstacle 
        for p1, p2 in zip(pts1, pts2):
            if p1[1] > 0 or p2[1] > 0: # bigger than floor 
                if p1[1] < y_thresh or p2[1] < y_thresh:
                    r_p1.append(p1)
                    r_p2.append(p2)
        r_pts1 = np.array(r_p1)
        r_pts2 = np.array(r_p2)
        
        return r_pts1, r_pts2
      
    def set_dt(self):
        img = np.ones_like(self.map_img) - self.map_img
        img[0, :] = 0 #TODO: move this to the original map img that I make
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0

        self.dt_img = ndimage.distance_transform_edt(img) * self.resolution
        self.dt_img = np.array(self.dt_img).T

        return self.dt_img

    def render_map(self, figure_n=1, wait=False):
        #TODO: draw the track boundaries nicely
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        plt.ylim([0, self.map_height])

        plt.imshow(self.map_img.T, origin='lower')

        xs = np.linspace(0, self.map_width, 10)
        ys = np.ones_like(xs) * self.end_y / self.resolution
        plt.plot(xs, ys, '--')     
        x, y = self.xy_to_row_column(self.start_pose[0:2])
        plt.plot(x, y, '*', markersize=14)

        plt.pause(0.0001)
        if wait:
            plt.show()
            pass

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resolution, 0, self.map_width-2)))
        r = int(round(np.clip(pt[1] / self.resolution, 0, self.map_height-2)))
        return c, r

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.forest_width or x_in[1] > self.forest_length:
            return True
        x, y = self.xy_to_row_column(x_in)
        if self.map_img[x, y]:
            return True

    def check_plan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.forest_width or x_in[1] > self.forest_length:
            return True
        x, y = self.xy_to_row_column(x_in)
        #TODO: figure out the x, y relationship
        # if self.dt_img[x, y] < 0.2:
        if self.dt_img[y, x] < 0.2:
            return True

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def render_wpts(self, wpts):
        plt.figure(4)
        xs, ys = self.convert_positions(wpts)
        plt.plot(xs, ys, '--', linewidth=2)
        # plt.plot(xs, ys, '+', markersize=12)

        plt.pause(0.0001)

    def render_aim_pts(self, pts):
        plt.figure(4)
        xs, ys = self.convert_positions(pts)
        # plt.plot(xs, ys, '--', linewidth=2)
        plt.plot(xs, ys, 'x', markersize=10)

        plt.pause(0.0001)

    def load_center_pts(self):

        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.ref_pts = track[:, 1:3]
        self.ss_normal = track[:, 0]
        # self.expand_wpts()
        # print(self.ref_pts)
        self.diffs = self.ref_pts[1:,:] - self.ref_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2



    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        new_line = []
        for i in range(len(self.wpts)-1):
            dd = sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

        self.wpts = np.array(new_line)



def load_conf(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return np.array(ret)

def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret


class ForestSim(BaseSim):
    """
    Simulator for Race Tracks

    Data members:
        map_name: name of the map to be used. Forest yaml file which stores the parameters for the forest. No image is required.

    """
    def __init__(self, sim_conf):
        """
        Init function

        Args:
            map_name: name of forest map to use.
            sim_conf: config file for simulation
        """

        env_map = ForestMap(sim_conf)
        BaseSim.__init__(self, env_map, self.check_done_forest, sim_conf
        )

    def check_done_forest(self):
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
        # horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # check forces
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # print(f"ThDot: {self.car.th_dot} --> Vel: {self.car.velocity}")
            # self.done_reason = f"Friction: {horizontal_force} > {self.car.max_friction_force}"

        # check steps
        elif self.steps > self.max_steps:
            self.done = True
            self.reward = -1
            self.done_reason = f"Max steps"
        # check orientation
        elif abs(self.state[2]) > 0.66*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1

        elif self.action[1] == 0 and self.state[3] < 1:
            self.done = True
            self.reward = -1
            self.done_reason = "Zero velocity"

        return self.done


    
