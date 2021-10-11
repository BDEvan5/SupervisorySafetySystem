import numpy as np
from numba import njit
from matplotlib import pyplot as plt

class Track:
    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.resolution = 20 # pts per meter
        self.grid = np.zeros((self.resolution*self.length, self.resolution*self.width))
        # self.grid = np.meshgrid([np.linspace(0, self.width, self.width*self.resolution), np.linspace(0, self.length, self.resolution*self.length)])


class State: 
    def __init__(self):
        self.X = None 
        self.Y = None
        self.phi = None 
        self.q = None 

        self.qs = None
        self.build_qs()

    def build_qs(self):
        v_max = 5  
        d_max = 0.4  
        L = 0.33
        n_vs = 5
        v_pts = np.linspace(1, v_max, n_vs)
        d_resolution = 0.1
        n_ds = [9, 9, 7, 5, 3] # number of steering points in triangle
        #TODO: automate this so that it can auto adjust
        n_modes = int(np.sum(n_ds))
        temp_qs = np.zeros((n_modes, 2))
        idx = 0
        for i in range(len(v_pts)): #step through vs
            for j in range(n_ds[i]): # step through ds 
                temp_qs[idx, 0] = (j - (n_ds[i] -1)/2)  * d_resolution
                temp_qs[idx, 1] = v_pts[i]
                idx += 1

        self.qs = np.zeros((n_modes, 2))
        for idx in range(n_modes):
            self.qs[idx, 1] = temp_qs[idx, 1]
            self.qs[idx, 0] = temp_qs[idx, 1] / L * np.tan(temp_qs[idx,0]) 

        plt.figure()
        for pt in self.qs:
        # for pt in temp_qs:
            plt.plot(pt[0], pt[1], 'ro')
        plt.show()


    def set_state(self, X, Y, phi, q):
        self.X = X
        self.Y = Y
        self.phi = phi
        self.q = q

    # def update_state(self, q):



if __name__ == "__main__":
    s = State()
    
