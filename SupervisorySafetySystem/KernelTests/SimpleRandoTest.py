from SupervisorySafetySystem.KernelTests.GeneralTestTrain import train_vehicle, test_single_vehicle
from SupervisorySafetySystem.NavAgents.SerialAgentPlanner import SerialVehicleTest, SerialVehicleTrain
from SupervisorySafetySystem.AgentKernel import AgentKernelTrain, AgentKernelTest

from SupervisorySafetySystem.KernelTests.ConstructingKernel import DiscriminatingImgKernel, Kernel, SafetyPlannerPP, KernelSim
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim

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
            s_p, r, done, _ = env.step(a)
            state = s_p

        if r == -1:
            print(f"{i}: Crashed -> {s_p['state']}")
        elif r == 1:
            print(f"{i}: Success")
            success += 1 

        if show:
            env.render_ep()
            plt.pause(0.5)

            if r == -1:
                plt.show()

    print("Success rate: {}".format(success/n_tests))

    return success/n_tests


def std_test():
    conf = load_conf("kernel_config")

    # constructy_kernel_sides(conf)
    # construct_obs_kernel(conf)

    env = KernelSim(conf)  
    planner = SafetyPlannerPP()
    planner.kernel = Kernel(conf)

    run_test_loop(env, planner, False, 10)


if __name__ == "__main__":
    std_test()

