from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator

import numpy as np
from matplotlib import pyplot as plt




def eval_magnitude_reward(sss_reward_scale, n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Mag_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = eval_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_constant_reward(sss_reward_scale, n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Const_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = eval_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def run_reward_tests():
    # n = 1
    for n in range(5):
        eval_constant_reward(0, n)
        eval_constant_reward(0.2, n)
        eval_constant_reward(0.5, n)
        eval_constant_reward(0.7, n)
        eval_constant_reward(1, n)

        eval_magnitude_reward(0.2, n)
        eval_magnitude_reward(0.5, n)
        eval_magnitude_reward(0.7, n)
        eval_magnitude_reward(1, n)


def learning_comparision_sss(n):
    sim_conf = load_conf("track_kernel")
    eval_name = f"LearningComparison_SSS_{n}"
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    agent_name = f"Kernel_Const_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Const_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)    

    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Mag_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Mag_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)


    test.run_free_eval(env, 100, wait=False)

def learning_comparision_pure(n):
    sim_conf = load_conf("track_kernel")
    eval_name = f"LearningComparison_pure_{n}"
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    agent_name = f"Kernel_Const_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Const_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)    

    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Mag_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Mag_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)


    test.run_free_eval(env, 100, wait=False)

def test_zero_vehicle(n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    test_kernel_vehicle(env, safety_planner, False, 100)
    # test_kernel_vehicle(env, safety_planner, True, 30)

def render_picture(n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    # agent_name = f"Kernel_Const_{0}_{n}"
    # agent_name = f"Kernel_Const_{1}_{n}"
    # agent_name = f"Kernel_Mag_{1}_{n}"
    agent_name = f"Kernel_Mag_NoTrain_{n}"
    # planner = EndVehicleTest(agent_name, sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    sim_conf.test_n = 4

    eval_kernel(env, safety_planner, sim_conf, False)
    # test_kernel_vehicle(env, safety_planner, True, 30)

if __name__ == "__main__":

    # run_reward_tests()
    n = 2
    # for n in range(5):
    #     learning_comparision_sss(n)
    #     learning_comparision_pure(n)
    # # learning_comparision_sss(n)
    # learning_comparision_pure(n)

    # test_zero_vehicle(n)

    render_picture(n)

