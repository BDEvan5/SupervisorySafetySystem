from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor


import numpy as np
from matplotlib import pyplot as plt



run_n = 1
baseline_name = f"std_end_baseline_{run_n}"

sim_conf = load_conf("track_kernel")


def tune_kernel_learn(k_learn_name, LearnReward, sss_reward_scale):
    test_name = f"k_learn{run_n}"
    agent_name = f"bTuning_{test_name}_{k_learn_name}_{run_n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = SteeringReward(0.05)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = LearnReward

    train_time = train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, True)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = train_time
    config_dict['test_number'] = run_n
    config_dict['reward'] = "Steer_05"
    config_dict['kernel_reward'] = k_learn_name
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_kernel_learn(k_learn_name):
    test_name = f"k_learn{run_n}"
    agent_name = f"bTuning_{test_name}_{k_learn_name}_{run_n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, True)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = 0
    config_dict['test_number'] = run_n
    config_dict['reward'] = "Steer_05"
    config_dict['kernel_reward'] = k_learn_name
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def magnitude_reward():
    k_learn_name = f"Magnitude_05"
    reward = MagnitudeReward(0.5)

    tune_kernel_learn(k_learn_name, reward, 0.5)


def zero_reward():
    k_learn_name = f"Zero"
    reward = MagnitudeReward(1)

    tune_kernel_learn(k_learn_name, reward, 0)


def constant_reward():
    k_learn_name = f"Constant_1"
    reward = MagnitudeReward(1)

    tune_kernel_learn(k_learn_name, reward, 1)





if __name__ == "__main__":
    magnitude_reward()
    zero_reward()
    constant_reward()
