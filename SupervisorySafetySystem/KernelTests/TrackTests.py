from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf

# from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SafetyWrapper import SafetyWrapper
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit, KernelTrackPP
from SupervisorySafetySystem.KernelGenerator import construct_obs_kernel, construct_kernel_sides

import yaml
from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt

test_n = 100



# def rando_test():
#     conf = load_conf("kernel_config")

#     construct_obs_kernel(conf)
#     construct_kernel_sides(conf)

#     env = TrackSim(conf)
#     planner = RandomPlanner()
#     safety_planner = SafetyWrapper(planner, conf)

#     test_kernel_vehicle(env, safety_planner, True, 10)


def pp_test():
    conf = load_conf("track_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = TrackSim(conf)
    planner = KernelTrackPP(conf)
    # safety_planner = SafetyWrapper(planner, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, planner, True, 10, add_obs=False)
    # test_kernel_vehicle(env, safety_planner, True, 10)

if __name__ == "__main__":
    # rando_test()
    pp_test()


