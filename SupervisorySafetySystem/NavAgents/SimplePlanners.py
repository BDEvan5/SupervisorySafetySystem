import numpy as np
from SupervisorySafetySystem.NavUtils import pure_pursuit_utils
from SupervisorySafetySystem.NavUtils.speed_utils import calculate_speed
import csv 
from SupervisorySafetySystem import LibFunctions as lib

from SupervisorySafetySystem.NavAgents.TrackPP import PurePursuit as TrackPP

from matplotlib import pyplot as plt
import os, shutil
from SupervisorySafetySystem.NavUtils.Trajectory import Trajectory

class RandomPlanner:
    def __init__(self, name="RandoPlanner"):
        self.d_max = 0.4 # radians  
        self.v = 2        
        self.name = name
        
        path = os.getcwd() + "/PaperData/Vehicles/" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path
        np.random.seed(1)


    def plan_act(self, obs):
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        v = calculate_speed(steering)
        return np.array([steering, v])

class ConstantPlanner:
    def __init__(self, name="StraightPlanner", value=0):
        self.steering_value = value
        self.v = 4        
        self.name = name

        path = os.getcwd() + "/PaperData/Vehicles/" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path


    def plan_act(self, obs):
        return np.array([self.steering_value, self.v])



class PurePursuit:
    def __init__(self, conf):
        self.name = "PurePursuit"
        

        self.trajectory = Trajectory(conf.map_name)

        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer

        self.progresses = []
        self.aim_pts = []

    def plan_act(self, obs):
        state = obs['state']
        pose_th = state[2]
        pos = state[0:2]
        v_current = state[3]

        self.progresses.append(obs['target'][1])

        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([0, 7])

        lookahead_point = self.trajectory.get_current_waypoint(pos, self.lookahead)
        self.aim_pts.append(lookahead_point)

        speed, steering_angle = pp_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        speed *= self.vgain

        # speed = calculate_speed(steering_angle)

        return np.array([steering_angle, speed])

    def plot_progress(self):
        plt.figure(2)
        plt.clf()
        # plt.plot(self.progresses)
        aim_pts = np.array(self.aim_pts)
        plt.plot(aim_pts[:, 0], aim_pts[:, 1])

        # plt.show()
        plt.pause(0.0001)

        # self.aim_pts.clear()

