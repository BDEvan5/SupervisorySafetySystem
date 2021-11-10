from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavUtils.RewardFunctions import *


import numpy as np
from matplotlib import pyplot as plt



run_n = 1
baseline_name = f"std_end_baseline_{run_n}"

sim_conf = load_conf("track_kernel")


def tune_baseline_reward(reward_name, Reward):
    test_name = f"reward_{run_n}"
    agent_name = f"bTuning_{test_name}_{reward_name}_{run_n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = Reward

    train_time = train_vehicle(env, planner, sim_conf)

    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, True)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = train_time
    config_dict['test_number'] = run_n
    config_dict['reward'] = reward_name
    # config_dict['b_heading'] = 0.004
    # config_dict['b_vel'] = 0.004
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_b_reward(reward_name):
    test_name = f"reward_{run_n}"
    agent_name = f"bTuning_{test_name}_{reward_name}_{run_n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, True)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = 0
    config_dict['test_number'] = run_n
    config_dict['reward'] = reward_name
    # config_dict['b_heading'] = 0.004
    # config_dict['b_vel'] = 0.004
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)



def cth_reward():
    reward_name = f"CthRef_1"
    reward_class = RefCTHReward(sim_conf, 0.04, 0.004)

    tune_baseline_reward(reward_name, reward_class)

def dist_reward():
    reward_name = f"DistRef_1"
    reward_class = DistReward()

    # tune_baseline_reward(reward_name, reward_class)
    eval_b_reward(reward_name)
    
def steer_reward():
    reward_name = f"Steer_1"
    reward_class = SteeringReward(0.1)

    tune_baseline_reward(reward_name, reward_class)
    

if __name__ == "__main__":
    # cth_reward()

    # dist_reward()
    steer_reward()

