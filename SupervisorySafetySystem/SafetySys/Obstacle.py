
import numpy as np
from matplotlib import pyplot as plt


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
