import numpy as np
from numba import njit 
import matplotlib.pyplot as plt

from SupervisorySafetySystem.SafetySys.safety_utils import *

def segment_lidar_scan(scan):
    """
    Take a lidar scan and return a set of critical x, y points
     """
    pass

def load_lidar():
    scan = np.load("lidar_scan.npy")
    return scan

# @njit(cache=True)
def segment_lidar_scan(scan):
    """ 
    Takes a lidar scan and reduces it to a set of points that make straight lines 
    TODO: possibly change implmentation to work completely in r, ths 
    """
    xs, ys = convert_scan_xy(scan)
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
    scaled_diffs = diffs / scan[:-1]

    i_pts = [0]
    d_thresh = 0.08
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

def segment_lidar_scan_scale(scan):
    """ 
    Takes a lidar scan and reduces it to a set of points that make straight lines 
    TODO: possibly change implmentation to work completely in r, ths 
    """
    xs, ys = convert_scan_xy(scan)
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)
    scaled_diffs = diffs / scan[:-1]

    i_pts = [0]
    d_thresh = 0.05
    for i in range(len(diffs)):
        if scaled_diffs[i] > d_thresh and diffs[i] > 0.08:
        # if scaled_diffs[i] > d_thresh:
            i_pts.append(i)
            i_pts.append(i+1)
    i_pts.append(len(scan)-1)
        
    i_pts = np.array(i_pts)
    x_pts = xs[i_pts]
    y_pts = ys[i_pts]

    return x_pts, y_pts

@njit(cache=True)
def convert_scan_xy(scan):
    sines, cosines = get_trigs(len(scan))
    xs = scan * sines
    ys = scan * cosines    
    return xs, ys




def run_lidar_process():
    scan = load_lidar()

    xs, ys = convert_scan_xy(scan)
    x_ints = np.arange(-500, 500)
    diffs = np.sqrt((xs[1:]-xs[:-1])**2 + (ys[1:]-ys[:-1])**2)

    # plt.figure(2)
    # plt.title("Range Lengths")
    # plt.plot(x_ints, scan)

    # plt.figure(3)
    # plt.title("Distances")
    # plt.plot(x_ints[:-1], diffs)
    # plt.ylim([0, 0.2])

    plt.figure(1)
    plt.title("Original Scan")
    plt.plot(xs, ys)

    # plt.figure(4)
    # plt.title("Segmented Scan")
    x_pts, y_pts = segment_lidar_scan(scan)
    plt.plot(x_pts, y_pts, 'x')

    x_pts_scale, y_pts_scale = segment_lidar_scan_scale(scan)
    plt.plot(x_pts_scale, y_pts_scale, '-+', markersize=20)

    # plt.figure(4)
    # plt.title("Scaled Distances")
    # scaled_diffs = diffs / scan[:-1]
    # plt.plot(x_ints[:-1], scaled_diffs)
    # plt.ylim([0, 0.2])


    plt.show()


if __name__ == "__main__":
    run_lidar_process()
