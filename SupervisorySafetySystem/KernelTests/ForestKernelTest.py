from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf

from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
# from SupervisorySafetySystem.ForestKernel import construct_obs_kernel, construct_kernel_sides
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit 
from SupervisorySafetySystem.SupervisorySystem import Supervisor, ForestKernel

import yaml
from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"
kernel_name = f"kernel_sap_{run_n}"



def rando_test():
    conf = load_conf("forest_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = RandomPlanner()
    kernel = ForestKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 20, wait=False)
    # test_kernel_vehicle(env, safety_planner, False, 100, wait=False)

def pp_test():
    conf = load_conf("forest_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    kernel = ForestKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, True, 10, wait=True)

if __name__ == "__main__":
    rando_test()
    # pp_test()