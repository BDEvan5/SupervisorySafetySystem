from matplotlib import pyplot as plt
import numpy as np

from SupervisorySafetySystem.KernelTests.GeneralTestTrain import test_kernel_vehicle, load_conf
from SupervisorySafetySystem.KernelGenerator import construct_obs_kernel, construct_kernel_sides
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
from SupervisorySafetySystem.NavAgents.SimplePlanners import PurePursuit
from SupervisorySafetySystem.SafetyWrapper import SafetyWrapper

def std_test():
    conf = load_conf("kernel_config")

    construct_obs_kernel(conf)
    # construct_kernel_sides(conf)

    env = ForestSim(conf)
    planner = PurePursuit(conf)
    safety_planner = SafetyWrapper(planner, conf)

    test_kernel_vehicle(env, safety_planner, True, 10)

    
def disretization_test():
    resolutions = [50, 80, 100, 120]
    results = np.zeros_like(resolutions)
    for i, resolution in enumerate(resolutions):
        print(f"Running discretisation test: {resolution}")
        side_name = f"discret_side_kernel_{resolution}"
        obs_name = f"discret_obs_kernel_{resolution}"
        constructy_kernel_sides(side_name, resolution)
        construct_obs_kernel(obs_name, resolution)

        env = KernelSim()  
        planner = SafetyPlannerPP()
        planner.kernel = Kernel(side_name, obs_name, resolution)

        results[i] = run_test_loop(env, planner, False, 10)

    print(resolutions)
    print(results)  

if __name__ == "__main__":
    std_test()
    # disretization_test()


