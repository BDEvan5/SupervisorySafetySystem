from GeneralTestTrain import test_kernel_vehicle, load_conf, test_normal_vehicle

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit, ConstantPlanner

import numpy as np
from matplotlib import pyplot as plt
from SupervisorySafetySystem.logger import LinkyLogger

def pp_kernel_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()

    env = TrackSim(conf)
    planner = PurePursuit(conf)
    kernel = TrackKernel(conf)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 1, add_obs=False, wait=True)


def pp_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()
    planner = PurePursuit(conf)
    link = LinkyLogger(conf, planner.name)
    env = TrackSim(conf, link)
    # kernel = TrackKernel(conf)
    # safety_planner = Supervisor(planner, kernel, conf)

    test_normal_vehicle(env, planner, True, 1, add_obs=False, wait=True)

def rando_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()

    link = LinkyLogger(conf, "RandoTest")
    env = TrackSim(conf, link)
    planner = RandomPlanner(conf, "RandoTest")
    
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 30, add_obs=False, wait=False)
    # test_kernel_vehicle(env, safety_planner, True, 30, add_obs=False, wait=True)
    # test_kernel_vehicle(env, safety_planner, True, 100, add_obs=False)
    # test_kernel_vehicle(env, safety_planner, False, 100, add_obs=False)


def straight_test():
    conf = load_conf("std_test_kernel")

    # build_track_kernel()
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(conf, link)
    # planner = ConstantPlanner("Left", -0.4)
    # planner = ConstantPlanner("Right", -0.4)
    planner = ConstantPlanner()
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    test_kernel_vehicle(env, safety_planner, True, 1, add_obs=False, wait=True)
  
def profile():
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    rando_test()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    # rando_test()
    # pp_kernel_test()
    pp_test()
    # straight_test()

    # profile()

