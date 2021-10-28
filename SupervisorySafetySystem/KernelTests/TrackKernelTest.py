from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit, EmptyPlanner
from SupervisorySafetySystem.NavAgents.TrackPP import PurePursuit as TrackPP
from SupervisorySafetySystem.StdTrackKernel import build_track_kernel

import numpy as np
from matplotlib import pyplot as plt


def pp_kernel_test():
    conf = load_conf("track_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    planner = TrackPP(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 10, add_obs=False)

def rando_test():
    conf = load_conf("track_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    planner = RandomPlanner()
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 30, add_obs=False)
    # test_kernel_vehicle(env, safety_planner, True, 100, add_obs=False)
    # test_kernel_vehicle(env, safety_planner, False, 100, add_obs=False)

if __name__ == "__main__":
    rando_test()
    # pp_kernel_test()

