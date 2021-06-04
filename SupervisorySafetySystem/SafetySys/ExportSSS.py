
from LearningLocalPlanning.NavUtils.SSS_utils import plot_safety_scan
import numpy as np
from numba import njit, jit 
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt as edt


def run_safety_check(obs, pp_action, max_steer, max_d_dot):
    scan = obs['full_scan']
    state = obs['state']

    dw_ds = build_dynamic_window(state[4], max_steer, max_d_dot, 0.1)

    valid_window, starts, ends = check_dw_vo(scan, dw_ds)

    if not valid_window.any():
        print(f"Massive problem: no valid answers")
        # plot_safety_scan(scan, starts, ends, dw_ds, valid_window, pp_action, pp_action)
        # plt.show()
        return pp_action, -1
    
    valid_dt = edt(valid_window)
    new_action = modify_action(pp_action, valid_window, dw_ds, valid_dt)

    # plot_safety_scan(scan, starts, ends, dw_ds, valid_window, pp_action, new_action)

    modified_flag = 0
    if new_action[0] != pp_action[0]: 
        modified_flag = -0.4 

    return new_action, modified_flag

@njit(cache=True) 
def build_dynamic_window(delta, max_steer, max_d_dot, dt):
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    n_delta_pts = 50 
    ds = np.linspace(ldb, udb, n_delta_pts)

    return ds

@njit(cache=True) 
def check_dw_vo(scan, dw_ds):
    # tuneable parameters
    d_cone = 1.6
    angle_buffer = 0.06
    L = 0.33

    angles = np.arange(1000) * np.pi / 999 -  np.ones(1000) * np.pi/2 

    valid_ds = np.ones_like(dw_ds)
    inds = np.arange(1000)

    invalids = inds[scan<d_cone]
    starts1 = invalids[1:][invalids[1:] != invalids[:-1] + 1]
    starts = np.concatenate((np.zeros(1, dtype=np.uint8), starts1))
    ends1 = invalids[:-1][invalids[1:] != invalids[:-1] + 1]
    ends = np.append(ends1, invalids[-1])

    for i in range(len(starts)):
    # for i, s, e in enumerate(zip(starts, ends)):
        # if i == 0 or i == len(starts) -1:
        #     continue
        s = int(starts[i])
        e = int(ends[i])
        d_min = np.arctan(2*L*np.sin(angles[s]-angle_buffer)/scan[s])
        d_max = np.arctan(2*L*np.sin(angles[e]+angle_buffer)/scan[e])

        d_min = max(d_min, dw_ds[0])
        d_max = min(d_max, dw_ds[-1]+0.001)

        i_min = np.count_nonzero(dw_ds[dw_ds<d_min])
        i_max = np.count_nonzero(dw_ds[dw_ds<d_max])

        valid_ds[i_min:i_max] = False

    return valid_ds, starts, ends
        

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
    window_sz = int(min(5, max(valid_dt)-1))
    for i in range(len(valid_window)): # search d space
        p_d = min(d_size-1, d_idx+i)
        if check_action_safe(valid_window, p_d, window_sz):
            return p_d 
        n_d = max(0, d_idx-i)
        if check_action_safe(valid_window, n_d, window_sz):
            return n_d 
    print(f"No Action Found: redo Search")



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


