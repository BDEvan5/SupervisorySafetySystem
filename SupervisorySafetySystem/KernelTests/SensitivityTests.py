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

def timestep_tests():
    # time_steps = [0.18, 0.2, 0.25, 0.3]
    time_steps = [0.1, 0.15, 0.18]
    results = np.zeros_like(time_steps)
    conf = load_conf("kernel_config")

    for i, time in enumerate(time_steps):
        print(f"Running discretisation test: {time}")
        
        conf.time_step = time
        conf.kernel_name = f"time_{time}"

        construct_obs_kernel(conf)
        construct_kernel_sides(conf)

        env = ForestSim(conf)
        planner = PurePursuit(conf)
        safety_planner = SafetyWrapper(planner, conf)

        results[i] = test_kernel_vehicle(env, safety_planner, True, 20)


    print(time_steps)
    print(results)  

def phi_tests():
    phis = [21, 41, 61, 81]
    results = np.zeros_like(phis)
    conf = load_conf("kernel_config")

    for i, n_phi in enumerate(phis):
        print(f"Running Phi test: {n_phi}")
        
        conf.n_phi = n_phi
        conf.kernel_name = f"phi_{n_phi}"

        construct_obs_kernel(conf)
        construct_kernel_sides(conf)

        env = ForestSim(conf)
        planner = PurePursuit(conf)
        safety_planner = SafetyWrapper(planner, conf)

        results[i] = test_kernel_vehicle(env, safety_planner, True, 20)


    print(phi_tests)
    print(results)  


if __name__ == "__main__":
    # std_test()
    # disretization_test()
    timestep_tests()
    # phi_tests()


