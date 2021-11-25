from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator

import numpy as np
from matplotlib import pyplot as plt


def eval_episodic(n):
    sss_reward_scale = 1
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Episodic_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RefCTHReward(sim_conf, 0.04, 0.004) 
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
    config_dict['learning'] = "Episodic"
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def eval_test_episodic(n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Episodic_{n}"

    planner = EndVehicleTest(agent_name, sim_conf)
    # safety_planner = Supervisor(planner, kernel, sim_conf)

    # eval_dict = eval_vehicle(env, planner, sim_conf, False)
    sim_conf.test_n = 5
    eval_dict = render_baseline(env, planner, sim_conf, False)
    
    # config_dict = vars(sim_conf)
    # config_dict['EvalName'] = "PaperTest" 
    # config_dict['test_number'] = n
    # config_dict['train_time'] = train_time
    # config_dict['learning'] = "Episodic"
    # config_dict['kernel_reward'] = "Magnitude"
    # config_dict['sss_reward_scale'] = sss_reward_scale
    # config_dict.update(eval_dict)

    # save_conf_dict(config_dict)


def eval_continuous(n):
    sss_reward_scale = 1
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Continuous_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RefCTHRewardContinuous(sim_conf, 0.04, 0.004) 
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_continuous_kernel(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = eval_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['learning'] = "Continuous"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

if __name__ == "__main__":
    # eval_continuous(6)
    # eval_episodic(2)
    eval_test_episodic(2)