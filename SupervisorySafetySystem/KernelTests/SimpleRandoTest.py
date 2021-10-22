from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle

from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
from SupervisorySafetySystem.SafetyWrapper import SafetyWrapper
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, PurePursuit 
from SupervisorySafetySystem.KernelGenerator import construct_obs_kernel, construct_kernel_sides

import yaml
from argparse import Namespace
import numpy as np
from matplotlib import pyplot as plt

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"
kernel_name = f"kernel_sap_{run_n}"

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


def run_test_loop(env, planner, show=False, n_tests=100):
    success = 0

    for i in range(n_tests):
        done = False
        state = env.reset()
        planner.kernel.construct_kernel(env.env_map.map_img.shape, env.env_map.obs_pts)
        while not done:
            a = planner.plan(state)
            s_p, r, done, _ = env.step_plan(a)
            state = s_p

        if r == -1:
            print(f"{i}: Crashed -> {s_p['state']}")
        elif r == 1:
            print(f"{i}: Success")
            success += 1 

        if show:
            env.render()
            plt.pause(0.5)

            if r == -1:
                plt.show()

    print("Success rate: {}".format(success/n_tests))

    return success/n_tests


def rando_test():
    conf = load_conf("kernel_config")

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = RandomPlanner()
    safety_planner = SafetyWrapper(planner, conf)

    run_test_loop(env, safety_planner, True, 10)

def pp_test():
    conf = load_conf("kernel_config")

    # construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    safety_planner = SafetyWrapper(planner, conf)

    # run_test_loop(env, safety_planner, True, 10)
    test_kernel_vehicle(env, safety_planner, True, 10)

if __name__ == "__main__":
    # rando_test()
    pp_test()