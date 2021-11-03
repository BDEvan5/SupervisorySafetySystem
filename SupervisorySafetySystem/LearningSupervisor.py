from SupervisorySafetySystem.SupervisorySystem import Supervisor 

import numpy as np
from matplotlib import pyplot as plt


class LearningSupervisor(Supervisor):
    def __init__(self, planner, kernel, conf):
        Supervisor.__init__(self, planner, kernel, conf)

        self.intervene = False


    def calculate_reward(self):
        if self.intervene:
            self.intervene = False
            return -1
        return 0

    def done_entry(self, s_prime):
        s_prime['reward'] = self.calculate_reward()
        self.planner.done_entry(s_prime)

