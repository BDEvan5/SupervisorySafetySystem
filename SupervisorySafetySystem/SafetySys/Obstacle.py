
import numpy as np
from matplotlib import pyplot as plt

class OrientationObstacle:
    def __init__(self, start, end):
        buffer = 0.05 
        self.start = start
        self.end = end

        self.start[0] -= buffer
        self.end[0] += buffer

    def run_check(self, state):
        pt = state[0:2]
        theta = state[2] + state[4]

        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        t_start = rot_m @ self.start 
        t_end = rot_m @ self.end  
        new_pt = rot_m @pt 
        
        # run checks
        if new_pt[0] < t_start[0] or new_pt[0] > t_end[0]:
            # print(f"Definitely safe: x outside range")
            safe_value = True
            return safe_value
            
        if new_pt[1] > t_start[1] and new_pt[1] > t_end[1]:
            # print(f"Definite crash, y is too big")
            safe_value =  False
            return safe_value
            
        d_min, d_max = get_d_lims(state[4])
        y_required = find_critical_point(new_pt[0], t_start, t_end, state[4])

        if y_required > new_pt[1]:
            safe_value = True 
        else:
            safe_value = False

        print(f"{safe_value} -> y_req:{y_required:.4f}, NewPt: {new_pt} ->start:{t_start}, end: {t_end}")
            
        plot_orig(t_start, t_end, new_pt, 0)
        xs = np.linspace(t_start[0]+0.01, t_end[0]-0.01, 15)
        ys = np.zeros_like(xs)
        for i in range(len(xs)):
            ys[i] = find_critical_point(xs[i], t_start, t_end, state[4])

        c_x = t_start[0] + (t_end[0] - t_start[0]) / 2
        plt.plot(c_x, np.mean((t_start[1], t_end[1])), 'X', markersize=10)
        
        plt.plot(xs, ys, '+-')
        # print(f"ys: {ys}")
        plt.pause(0.0001)

        return safe_value


def find_critical_point(x, start, end, current_d):
    if x < start[0] or x > end[0]:
        print(f"Excluding obstacle")
        return 0 #

    c_x = start[0] + (end[0] - start[0]) / 2

    sv = 3.2 
    speed = 3 

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
    plt.figure(3)
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    plt.plot(xs, ys)
        
    plt.arrow(pt[0], pt[1], np.sin(theta)*0.1, np.cos(theta)*0.1, width=0.01, head_width=0.03)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)

def plot_obs(p1, p2, pt, pl, pr):
    plt.figure(3)
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    plt.plot(xs, ys)
        
    plt.arrow(pt[0], pt[1], 0, 0.1, width=0.01, head_width=0.03)

    pts = np.vstack((p1, p2, pr, pl, p1))
    plt.plot(pts[:, 0], pts[:, 1])
    plt.plot(0, 0, 'x', markersize=20)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)

        
def get_d_lims(d, t=0.1):
    sv = 3.2
    d_min = max(d-sv*t, -0.4)
    d_max = min(d+sv*t, 0.4)
    return d_min, d_max
    



def y_interpolation(A, B, x_val):
    return A[1] + (B[1]-A[1]) * (x_val-A[0]) / (B[0]-A[0])


def test_orientation():
    p1 = [-0.2, 0.5]
    p2 = [0.3, 0.6]
    o = OrientationObstacle(p1, p2)

    # o.check_location_safe([0.1, 0.1], 0)
    o.run_check([0.2, 0.1], -0.4, -0.4, 0.4)
    o.draw_obstacle()
    o.plot_arc()
    plt.show()
    o.run_check([-0.2, 0.1], 0.6, -0.4, 0.4)
    o.draw_obstacle()
    o.plot_arc()
    plt.show()

if __name__ == '__main__':

    test_orientation()
