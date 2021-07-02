import numpy as np
import matplotlib.pyplot as plt
from SupervisorySafetySystem.test_dyns import control_system, update_kinematic_state
from numba import jit, njit
from scipy.ndimage import distance_transform_edt as edt



class OrientationObstacle:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def plot_obstacle(self):
        pts = np.vstack((self.start, self.end))
        plt.plot(pts[:, 0], pts[:, 1], '-x', markersize=16)

    def run_check(self, state):
        pt = state[0:2]
        theta = state[2]#+ state[4]

        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        t_start = rot_m @ self.start 
        t_end = rot_m @ self.end  
        new_pt = rot_m @pt 

        plot_orig(t_start, t_end, new_pt, 0)
        
        # run checks
        if new_pt[0] < t_start[0] or new_pt[0] > t_end[0]:
            print(f"Definitely safe: x outside range")
            safe_value = True
            return safe_value
            
        if new_pt[1] > t_start[1] and new_pt[1] > t_end[1]:
            print(f"Definite crash, y is too big")
            safe_value =  False
            return safe_value
            
        # d_min, d_max = get_d_lims(state[4])
        y_required = find_critical_point(new_pt[0], t_start, t_end, state[4])

        if y_required > new_pt[1]:
            safe_value = True 
        else:
            safe_value = False

        return safe_value


def find_critical_point(x, start, end, current_d, speed=1):
    if x < start[0] or x > end[0]:
        print(f"Excluding obstacle")
        return 0 #

    c_x = start[0] + (end[0] - start[0]) / 2

    sv = 3.2 
    # speed = 3 

    if x > c_x:
        width = end[0] - x
        extra_d = (0.4-current_d) / sv * speed
        d_required = find_distance_obs(width, 0.4) + extra_d
        critical_y = end[1] - d_required

    else:
        width = x - start[0]
        extra_d = current_d + 0.4 / sv * speed
        d_required = find_distance_obs(width, 0.4) + extra_d
        critical_y = start[1] - d_required

    return critical_y


def find_distance_obs(w, d_max=0.4, L=0.33):
    ld = np.sqrt(w*2*L/np.tan(d_max))
    distance = (ld**2 - (w**2))**0.5
    return distance

def plot_orig(p1, p2, pt, theta):
    plt.figure(2)
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    plt.plot(xs, ys)
        
    plt.arrow(pt[0], pt[1], np.sin(theta)*0.1, np.cos(theta)*0.1, width=0.01, head_width=0.03)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)


def run_obs_avoid():
    state = np.array([0, 0, 0, 1, 0])

    start = [-0.3, 0.8]
    end = [0.3, 0.8]
    obs = OrientationObstacle(start, end)

    state_history = []
    
    for i in range(20):
        state_history.append(state[0:2])
        action = run_avoid_system(state, obs, state_history)
        state = run_step(state, action) 

        print(f"{i} -> State: {state}")

# @jit
def run_step(x, a, n_steps=1):
    for _ in range(10):
        u = control_system(x, a, 7, 0.4, 8, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.329, 0.4, 7)

    return x 

def run_avoid_system(state, obs, state_history):
    dw_ds = build_dynamic_window(state[4], n_pts=5)

    next_states = simulate_sampled_actions(dw_ds, state)

    obstacles = [obs]
    valid_window = classify_next_states(next_states, obstacles, state)


    valid_dt = edt(valid_window)
    new_action = modify_action(np.array([0, 1], dtype=np.float64), valid_window, dw_ds, valid_dt)

    plot_picture(state, next_states, obstacles, dw_ds, valid_window,new_action, np.array(state_history))

    return new_action
    
def plot_picture(state, next_states, obses, dw_ds, valid_window, action, state_history):
    plt.figure(1)
    plt.clf()
    
    plt.xlim([-1, 1])
    plt.ylim([0, 2])
    plt.title("Pretty Picture Single Obs")

    for obs in obses:
        obs.plot_obstacle()

    plt.text(-0.75, 1.8, f"Action: {action[0]}")

    for j, d in enumerate(dw_ds):
        if valid_window[j]:
            plt.plot(d, 1.5, 'x', color='green', markersize=14)
        else:
            plt.plot(d, 1.5, 'x', color='red', markersize=14)

    plt.plot(state[0], state[1], 'x', markersize=20)
    plt.plot(state_history[:, 0], state_history[:, 1], '-x')

    scale = 0.2
    for x_p in next_states:
        vx = [state[0],x_p[0]]
        vy = [state[1], x_p[1]]
        plt.plot(vx, vy, '--')
        plt.arrow(x_p[0], x_p[1], scale*np.sin(x_p[2]), scale*np.cos(x_p[2]), head_width=0.03)

    plt.show()

@njit(cache=True) 
def build_dynamic_window(delta, max_steer=0.4, max_d_dot=3.2, dt=0.1, n_pts=10):
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    return np.linspace(ldb, udb, n_pts)


# @jit(cache=True)
def simulate_sampled_actions(dw_ds, state):
    speed = max(state[3], 1)
    next_states = np.zeros((len(dw_ds), 5))
    x_state = np.array([0, 0, state[2], state[3], state[4]])
    for i, d in enumerate(dw_ds):
        action = np.array([d, speed])
        x_prime = run_step(x_state, action, 1)
        next_states[i] = x_prime

    return next_states
    
def classify_next_states(next_states, obstacles, state):
    n = len(next_states) 
    valid_ds = np.ones(n)
    for i in range(n):
        safe = True 
        next_states[i, 0:3] += state[0:3]
        for obs in obstacles:
            if not obs.run_check(next_states[i]):
                safe = False 
                break 
        valid_ds[i] = safe 

    return valid_ds 
    
        
# @jit(cache=True)
def modify_action(pp_action, valid_window, dw_ds, valid_dt):
    d_idx = np.count_nonzero(dw_ds[dw_ds<pp_action[0]])
    if check_action_safe(valid_window, d_idx):
        new_action = pp_action 
    else: 
        d_idx_search = np.argmin(np.abs(dw_ds))
        d_idx = int(find_new_action(valid_window, d_idx_search, valid_dt))
        new_action = np.array([dw_ds[d_idx], 1], dtype=np.float64)
        new_action = new_action
    return new_action

@jit(cache=True)
def find_new_action(valid_window, d_idx, valid_dt):
    d_size = len(valid_window)
    for i in range(len(valid_window)): # search d space
        n_d = max(0, d_idx-i)
        if check_action_safe(valid_window, n_d):
            return n_d 
        p_d = min(d_size-1, d_idx+i)
        if check_action_safe(valid_window, p_d):
            return p_d 
    # no valid window options, take only option left 
    return np.count_nonzero(valid_window)
    


@njit(cache=True)
def check_action_safe(valid_window, d_idx):
    if valid_window[d_idx]:
        return True
    return False
        
         



run_obs_avoid()



