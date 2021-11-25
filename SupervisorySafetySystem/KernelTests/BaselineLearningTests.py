from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator

import numpy as np
from matplotlib import pyplot as plt




def train_baseline_rewards(n):
    sim_conf = load_conf("track_kernel")
    test_name = "reward"
    test_value = 0.04
    # test_id = "Steer"
    # test_id = "cthRef"
    test_id = "cthCent"
    # test_id = "distRef"
    # test_id = "distCent"
    agent_name = f"Baseline_{test_name}_{test_id}_{test_value}_{n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    # planner.calculate_reward = SteeringReward(0.01) 
    # planner.calculate_reward = RefCTHReward(sim_conf, 0.04, 0.004) 
    planner.calculate_reward = CenterCTHReward(sim_conf, 0.04, 0.004) 
    # planner.calculate_reward = RefDistanceReward(sim_conf, 1) 
    # planner.calculate_reward = CenterDistanceReward(sim_conf, 1) 

    train_time, crashes = train_vehicle(env, planner, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['reward'] = test_id
    config_dict['r1'] = test_value
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def train_baseline_distance(n):
    sim_conf = load_conf("track_kernel")
    test_name = "reward"
    test_value = 1
    test_id = "distCent"
    agent_name = f"Baseline_{test_name}_{test_id}_{test_value}_{n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = CenterDistanceReward(sim_conf, 1) 

    train_time, crashes = train_vehicle(env, planner, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['reward'] = test_id
    config_dict['r1'] = test_value
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def train_baseline_cth(n):
    sim_conf = load_conf("track_kernel")
    test_name = "reward"
    test_value = 0.04
    test_id = "cthRef"
    agent_name = f"Baseline_{test_name}_{test_id}_{test_value}_{n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RefCTHReward(sim_conf, 0.04, 0.004) 

    train_time, crashes = train_vehicle(env, planner, sim_conf)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = eval_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['reward'] = test_id
    config_dict['r1'] = test_value
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def eval_baseline_cth(n):
    sim_conf = load_conf("track_kernel")
    test_name = "reward"
    test_value = 0.04
    test_id = "cthRef"
    agent_name = f"Baseline_{test_name}_{test_id}_{test_value}_{n}"

    env = TrackSim(sim_conf)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = eval_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    # config_dict['train_time'] = train_time
    config_dict['reward'] = test_id
    config_dict['r1'] = test_value
    # config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)



def render_picture(n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    # agent_name = f"Kernel_Const_{0}_{n}"
    # agent_name = f"Kernel_Const_{1}_{n}"
    agent_name = f"Baseline_reward_cthRef_0.04_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    sim_conf.test_n = 4

    # eval_kernel(env, safety_planner, sim_conf, False)
    render_baseline(env, planner, sim_conf, False)
    # test_kernel_vehicle(env, safety_planner, True, 30)



if __name__ == "__main__":
    # train_baseline_rewards(1)
    # train_baseline_distance(2)
    # train_baseline_cth(4)
    # eval_baseline_cth(1)

    render_picture(1)


