from GeneralTestTrain import test_kernel_vehicle, load_conf

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit, ConstantPlanner

import numpy as np
from matplotlib import pyplot as plt


def pp_kernel_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    planner = PurePursuit(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 1, add_obs=False, wait=True)

def rando_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    planner = RandomPlanner()
    kernel = TrackKernel(conf, False)
    # kernel = TrackKernel(conf, False, f"TrackKernel_{conf.track_kernel_path}_{conf.map_name}.npy")
    # kernel = TrackKernel(conf, False, f"DiscKern_{conf.track_kernel_path}_{conf.map_name}.npy")
    safety_planner = Supervisor(planner, kernel, conf)

    # test_kernel_vehicle(env, safety_planner, True, 30, add_obs=False, wait=False)
    # test_kernel_vehicle(env, safety_planner, True, 30, add_obs=False, wait=True)
    # test_kernel_vehicle(env, safety_planner, True, 100, add_obs=False)
    test_kernel_vehicle(env, safety_planner, False, 100, add_obs=False)


def straight_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    # planner = ConstantPlanner("Left", -0.4)
    # planner = ConstantPlanner("Right", -0.4)
    planner = ConstantPlanner()
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 1, add_obs=False, wait=True)
  

if __name__ == "__main__":
    rando_test()
    # pp_kernel_test()
    # straight_test()

