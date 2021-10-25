from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf

# from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.TrackWrapper import TrackWrapper
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit, KernelTrackPP
from SupervisorySafetySystem.NavAgents.TrackPP import PurePursuit as TrackPP

import yaml
from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt


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

def pp_kernel_test():
    conf = load_conf("track_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = TrackSim(conf)
    planner = TrackPP(conf)
    safety_planner = TrackWrapper(planner, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, True, 10, add_obs=False)

def rando_test():
    conf = load_conf("track_kernel")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = TrackSim(conf)
    planner = RandomPlanner()
    safety_planner = TrackWrapper(planner, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, True, 10, add_obs=False)

if __name__ == "__main__":
    rando_test()
    # pp_test()
    # pp_kernel_test()

