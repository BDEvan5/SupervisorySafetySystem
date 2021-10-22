from matplotlib import pyplot as plt
import numpy as np

from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf
from SupervisorySafetySystem.KernelGenerator import construct_obs_kernel, construct_kernel_sides
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
from SupervisorySafetySystem.NavAgents.SimplePlanners import PurePursuit
from SupervisorySafetySystem.SafetyWrapper import SafetyWrapper

def std_test():
    conf = load_conf("kernel_config")
    # conf.resolution = 0.0125

    construct_obs_kernel(conf)
    construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    safety_planner = SafetyWrapper(planner, conf)

    test_kernel_vehicle(env, safety_planner, True, 20)

    
def disretization_test():
    n_dxs = [80, 100, 120]
    results = np.zeros_like(n_dxs)
    conf = load_conf("kernel_config")

    for i, resolution in enumerate(n_dxs):
        print(f"Running discretisation test: {resolution}")
        
        conf.resolution = resolution
        conf.kernel_name = f"n_dx_{resolution}"

        construct_obs_kernel(conf)
        construct_kernel_sides(conf)

        env = ForestSim(conf)
        planner = PurePursuit(conf)
        safety_planner = SafetyWrapper(planner, conf)

        results[i] = test_kernel_vehicle(env, safety_planner, True, 20)


    print(n_dxs)
    print(results)  

if __name__ == "__main__":
    # std_test()
    disretization_test()


