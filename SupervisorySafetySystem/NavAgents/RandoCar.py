import numpy as np
from SupervisorySafetySystem.SafetySys.ExportSSS import run_safety_check

class RandoCar:
    def __init__(self, sim_conf):
        self.name = "Rando Car"
        self.max_steer = sim_conf.max_steer
        self.max_d_dot = sim_conf.max_d_dot

        np.random.seed(12345)

    def plan_act(self, obs):
        speed = 3
        steer = np.random.uniform(-self.max_steer, self.max_steer)
        action = np.array([steer, speed])

        new_action, _m = run_safety_check(obs, action, self.max_steer, self.max_d_dot)

        return new_action
