
import numpy as np
from matplotlib import pyplot as plt

class OrientationObstacle:
    def __init__(self, start, end):
        buffer = 0.05 
        self.start = start
        self.end = end

        self.start[0] -= buffer
        self.end[0] += buffer

    def check_location_safe(self, pt, theta):
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        t_start = rot_m @ self.start 
        t_end = rot_m @ self.end  

        new_pt = rot_m @pt 

        if new_pt[0] < t_start[0] or new_pt[0] > t_end[0]:
            # this obs isn't important then
            # pt is not in line
            print(f"Definitely safe: x outside range")
            return True
        
        if pt[1] > t_start[1] and pt[1] > t_end[1]:
            # definitely going to crash
            print(f"Definite crash, y is too big")
            return False

        c_x = (t_start[0] + t_end[0]) / 2
        if new_pt[0] > c_x:
            # right side 
            w_right = t_end[0] - new_pt[0]
            d_required = find_distance_obs(w_right)
            d_to_obs = t_end[1] - new_pt[1]
        elif new_pt[0] <= c_x:
            w_left = new_pt[0] - t_start[0]
            d_required = find_distance_obs(w_left)
            d_to_obs = t_start[1] - new_pt[1]
        else:
            raise NotImplementedError()

        if d_required < d_to_obs:
            ret_val = True # it is safe to continue
        else:
            ret_val = False

        print(f"Returning: {ret_val}")
        return ret_val
        
        
            
    def draw_triange(self, pt, theta):
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        t_start = rot_m @ self.start 
        t_end = rot_m @ self.end  

        new_pt = rot_m @pt 

        print(f"T start: {t_start}")
        print(f"T end: {t_end}")
        print(f"T pt: {new_pt}")

        c_x = (t_start[0] + t_end[0]) / 2
        w_right = t_end[0] - c_x
        d_required_left = find_distance_obs(w_right)
        w_left = c_x - t_start[0]
        d_required_right = find_distance_obs(w_left)

        pt_left = [c_x-0.01, t_start[1] - d_required_left]
        pt_right = [c_x+0.01, t_end[1] - d_required_right]


        plot_orig(self.start, self.end, pt, theta)
        plot_obs(t_start, t_end, new_pt, pt_left, pt_right)
        
        plt.show()


def plot_orig(p1, p2, pt, theta):
    plt.figure(1)
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    plt.plot(xs, ys)
        
    plt.arrow(pt[0], pt[1], np.sin(theta)*0.1, np.cos(theta)*0.1, width=0.01, head_width=0.03)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)

def plot_obs(p1, p2, pt, pl, pr):
    plt.figure(2)
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    plt.plot(xs, ys)
        
    plt.arrow(pt[0], pt[1], 0, 0.1, width=0.01, head_width=0.03)

    pts = np.vstack((p1, p2, pr, pl, p1))
    plt.plot(pts[:, 0], pts[:, 1])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.pause(0.0001)



class Obstacle:
    def __init__(self, start, end):
        buffer = 0.1 
        self.start_x = start[0] - buffer 
        self.start_y = start[1]
        self.end_x = end[0] + buffer
        self.end_y = end[1]

        c_x = (self.start_x + self.end_x) / 2
        self.c_x = c_x

        w_left = c_x - self.start_x 
        d_left = find_distance_obs(w_left)
        self.pt_left_center = [c_x-0.01, self.start_y-d_left]
        
        w_right = self.end_x - c_x 
        d_right = find_distance_obs(w_right)
        self.pt_right_center = [c_x+0.01,self.end_y-d_right]
        
    def check_location_safe(self, pt):
        x, y = pt
        if x < self.start_x or x > self.end_x:
            # obstacle is not considered 
            return True 
        
        if y > self.start_y and y > self.end_y:
            # it will definitely crash, in line with obs
            print(f"Definitely crashing: {y} too big")
            return False 

        # TODO: add check condition for if pt[1] < self.c_x[1, return true]

        if x > self.c_x:
            ret_val = self.check_right_side(pt) 
        elif x < self.c_x:
            ret_val = self.check_left_side(pt)
        else:
            ret_val = self.check_both_sides(pt)

        # if ret_val is False:
        #     print("Unsafe action: pt")

        return ret_val
        
    def check_right_side(self, pt):
        # x > cx and x < self.x_end
        y_interp = y_interpolation(self.pt_right_center, [self.end_x, self.end_y], pt[0])
        if y_interp > pt[1]:
            print(f"TRUE right: Pt val: {pt}, Interp: {y_interp:.4f}")
            return True
        print(f"FALSE right: Pt val: {pt}, Interp: {y_interp:.4f}")
        return False    
        
                
    def check_left_side(self, pt):
        # x < 0 and x > self.start_x 
        s_pt = [self.start_x, self.start_y]
        y_interp = y_interpolation(self.pt_left_center, s_pt, pt[0])
        if y_interp > pt[1]:
            print(f"TRUE left: Pt val: {pt}, Interp: {y_interp:.4f}-> s:{s_pt}")
            return True
        print(f"FALSE left: Pt val: {pt}, Interp: {y_interp:.4f} -> s:{s_pt}")
        return False    

    def check_both_sides(self, pt):
        y_interp_r = y_interpolation(self.pt_right_center, [self.end_x, self.end_y], pt[0])
        y_interp_l = y_interpolation(self.pt_left_center, [self.start_x, self.start_y], pt[0])
        if y_interp_r > pt[1] and y_interp_l > pt[1]:
            print(f"TRUE center: Pt val: {pt}, Interp: {y_interp_l:.4f}")
            return True
        print(f"FALSE center: Pt val: {pt}, Interp: {y_interp_l:.4f}")
        return False    
        
    def plot_obs_pts(self):
        if self.start_x > 0.8 or self.end_x < -0.8:
            return

        pt_left = [self.start_x, self.start_y]
        pt_right = [self.end_x, self.end_y]

        pts = np.vstack((pt_left,pt_right, self.pt_right_center, self.pt_left_center, pt_left))

        plt.plot(pts[:, 0], pts[:, 1])
        
def y_interpolation(A, B, x_val):
    return A[1] + (B[1]-A[1]) * (x_val-A[0]) / (B[0]-A[0])

def find_distance_obs(w, L=0.33, d_max=0.4):
    ld = np.sqrt(w*2*L/np.tan(d_max))
    distance = ((ld)**2 - (w**2))**0.5
    return distance


def test_orientation():
    p1 = [-0.2, 0.5]
    p2 = [0.3, 0.6]
    o = OrientationObstacle(p1, p2)

    # o.check_location_safe([0.1, 0.1], 0)
    o.check_location_safe([0.2, 0.1], -0.4)
    o.draw_triange([0.2, 0.1], -0.4)
    o.check_location_safe([-0.2, 0.1], 0.6)
    o.draw_triange([-0.2, 0.1], 0.6)


test_orientation()
