import numpy as np 
from numba import njit
 
from LearningLocalPlanning.NavUtils.speed_utils import calculate_speed
import os, shutil
from matplotlib import pyplot as plt

#TODO: most of this can be njitted
class ForestFGM:    
    BUBBLE_RADIUS = 400
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 100
    MAX_LIDAR_DIST = 10
    REDUCTION = 200
    
    def __init__(self, sim_conf, name="FGM"):
        self.degrees_per_elem = None
        self.name = name
        self.n_beams = sim_conf.n_beams
        self.max_steer = sim_conf.max_steer

        path = sim_conf.vehicle_path + name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path
    
    def preprocess_lidar(self, ranges):
        self.degrees_per_elem = (180) / len(ranges)
        proc_ranges = np.array(ranges[self.REDUCTION:-self.REDUCTION])
        # proc_ranges = ranges
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        return proc_ranges
   
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') 
        averaged_max_gap = averaged_max_gap / self.BEST_POINT_CONV_SIZE
        best = averaged_max_gap.argmax()
        idx = best + start_i

        return idx

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data
        """
        return (range_index - (range_len/2)) * self.degrees_per_elem *0.5

    def process_lidar(self, ranges):
        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        # self.vis.add_step(proc_ranges, steering_angle)
        self.plot_scan(ranges, proc_ranges, best)

        return steering_angle

    def plot_scan(self, ranges, proc_ranges, best):
        plt.figure(1)
        plt.clf()
        xs = ranges = np.cos(np.linspace(0, np.pi, len(ranges))) * ranges
        ys = ranges = np.sin(np.linspace(0, np.pi, len(ranges))) * ranges
        plt.plot(xs, ys, 'b')
        
        xs = proc_ranges = np.cos(np.linspace(0, np.pi, len(proc_ranges))) * proc_ranges
        ys = proc_ranges = np.sin(np.linspace(0, np.pi, len(proc_ranges))) * proc_ranges
        plt.plot(xs, ys, 'r')

        plt.pause(0.0001)

    def plan_act(self, obs):
        scan = obs['full_scan']
        ranges = np.array(scan, dtype=np.float)

        steering_angle = self.process_lidar(ranges)
        steering_angle = steering_angle * np.pi / 180
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        # speed = 4
        speed = calculate_speed(steering_angle) * 0.8

        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

# @njit
def find_max_gap(free_space_ranges):
    """ Return the start index & end index of the max gap in free_space_ranges
        free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
    """
    # mask the bubble
    masked = np.ma.masked_where(free_space_ranges==0, free_space_ranges)
    # get a slice for each contigous sequence of non-bubble data
    slices = np.ma.notmasked_contiguous(masked)
    if len(slices) == 0:
        return 0, len(free_space_ranges)
    max_len = slices[0].stop - slices[0].start
    chosen_slice = slices[0]
    # I think we will only ever have a maximum of 2 slices but will handle an
    # indefinitely sized list for portablility
    for sl in slices[1:]:
        sl_len = sl.stop - sl.start
        if sl_len > max_len:
            max_len = sl_len
            chosen_slice = sl
    return chosen_slice.start, chosen_slice.stop


