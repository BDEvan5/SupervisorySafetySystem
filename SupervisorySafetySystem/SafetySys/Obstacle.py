
import numpy as np
from matplotlib import pyplot as plt

class OrientationObstacle:
    def __init__(self, start, end):
        buffer = 0.05 
        self.start = start
        self.end = end

        self.start[0] -= buffer
        self.end[0] += buffer

        self.t_start = None 
        self.t_end = None
        self.new_pt = None
        self.c_x = None 
        self.pt_left = None
        self.pt_right = None
        self.d_min = None 
        self.d_max = None

    def reset_pts(self):
        self.t_start = [0, 0]
        self.t_end = [0, 0]
        self.new_pt = [0, 0]
        self.c_x = [0, 0] 
        self.pt_left = [0, 0]
        self.pt_right = [0, 0]
        
    def run_check(self, pt, theta, d_min, d_max):
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        self.t_start = rot_m @ self.start 
        self.t_end = rot_m @ self.end  
        self.new_pt = rot_m @pt 

        self.d_min = d_min
        self.d_max = d_max
        
        # run checks
        if self.new_pt[0] < self.t_start[0] or self.new_pt[0] > self.t_end[0]:
            # print(f"Definitely safe: x outside range")
            self.reset_pts()
            self.safe_value = True
            return
        
        if self.new_pt[1] > self.t_start[1] and self.new_pt[1] > self.t_end[1]:
            # print(f"Definite crash, y is too big")
            self.reset_pts()
            self.safe_value =  False
            return

        self.c_x = self.t_start[0] + (self.t_end[0] - self.t_start[0]) / 2
        
        y_required = self.find_critical_point(self.new_pt[0])

        if y_required > self.new_pt[1]:
            self.safe_value = True 
        else:
            self.safe_value = False

        print(f"{self.safe_value} -> y_req:{y_required:.4f}, NewPt: {self.new_pt} ->start:{self.t_start}, end: {self.t_end}")
            
        plot_orig(self.t_start, self.t_end, self.new_pt, 0)
        xs = np.linspace(self.t_start[0]+0.01, self.t_end[0]-0.01, 15)
        ys = np.zeros_like(xs)
        for i in range(len(xs)):
            ys[i] = self.find_critical_point(xs[i])
        
        plt.plot(xs, ys, '+-')
        # print(f"ys: {ys}")
        plt.pause(0.0001)

    def find_critical_point(self, x):
        if x < self.t_start[0] or x > self.t_end[0]:
            return 0 #

        if x > self.c_x:
            width = self.t_end[0] - x
            d_required = find_distance_obs(width, self.d_max)
            return self.t_end[1] - d_required

        else:
            width = x - self.t_start[0]
            d_required = find_distance_obs(width, abs(self.d_min))
            return self.t_start[1] - d_required


    def is_safe(self):
        return self.safe_value


    def plot_arc(self):
        xs = np.linspace(self.t_start[0]+0.01, self.t_end[0]-0.01, 30)
        ys = np.zeros_like(xs)
        for i in range(len(xs)):
            ys[i] = self.find_critical_point(xs[i])
        
        plt.plot(xs, ys, '+-')
        plt.pause(0.0001)


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




def y_interpolation(A, B, x_val):
    return A[1] + (B[1]-A[1]) * (x_val-A[0]) / (B[0]-A[0])

def find_distance_obs(w, d_max=0.4, L=0.33):
    ld = np.sqrt(w*2*L/np.tan(d_max))
    distance = (ld**2 - (w**2))**0.5
    return distance


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
