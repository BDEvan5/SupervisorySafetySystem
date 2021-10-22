import numpy as np


class RandomPlanner:
    def __init__(self):
        self.d_max = 0.4 # radians  
        self.v = 2        

    def plan_act(self, obs):
        np.random.seed()
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        return np.array([steering, self.v])



class PurePursuit:
    def __init__(self, sim_conf):
        self.name = "PurePursuit Planner"
        self.v = 2
        self.d_max= sim_conf.max_steer
        self.L = sim_conf.l_f + sim_conf.l_r
        self.lookahead_distance = 1
 
    def plan_act(self, obs):
        state = obs['state']
        pose_theta = state[2]
        lookahead = np.array([1, state[1]+self.lookahead_distance]) #pt 1 m in the future on centerline
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead[0:2]-state[0:2])
        if np.abs(waypoint_y) < 1e-6:
            return np.array([0, self.v])
        radius = 1/(2.0*waypoint_y/self.lookahead_distance**2)
        steering_angle = np.arctan(self.L/radius)
        steering_angle = np.clip(steering_angle, -self.d_max, self.d_max)
        return np.array([steering_angle, self.v])
   

