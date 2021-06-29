import numpy as np
from numba import njit


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
