from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavUtils.RewardFunctions import *


import numpy as np
from matplotlib import pyplot as plt



run_n = 2

sim_conf = load_conf("track_kernel")


def tune_baseline_reward(reward_name, Reward, r1=0, r2=0):
    test_name = f"reward"
    agent_name = f"bTuning_{test_name}_{reward_name}_{r1}_{run_n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = Reward

    train_time = train_vehicle(env, planner, sim_conf)

    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = eval_vehicle(env, planner, sim_conf, False)

    config_dict = vars(sim_conf)
    config_dict['EvalName'] = test_name 
    config_dict['train_time'] = train_time
    config_dict['test_number'] = run_n
    config_dict['reward'] = reward_name
    config_dict['r1'] = r1
    config_dict['r2'] = r2
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_b_reward(reward_name, r1=0, r2=0):
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
    config_dict['r1'] = r1
    config_dict['r2'] = r2
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)



def cth_reward():
    reward_name = f"CthRef"
    r1 = 0.03
    r2 = 0.004
    reward_class = RefCTHReward(sim_conf, r1, r2)

    tune_baseline_reward(reward_name, reward_class, r1, r2)

    reward_name = f"CthCenter"
    r1 = 0.03
    r2 = 0.004
    reward_class = CenterCTHReward(sim_conf, r1, r2)

    tune_baseline_reward(reward_name, reward_class, r1, r2)

    reward_name = f"CthRef"
    r1 = 0.02
    r2 = 0.004
    reward_class = RefCTHReward(sim_conf, r1, r2)

    tune_baseline_reward(reward_name, reward_class, r1, r2)

    reward_name = f"CthCenter"
    r1 = 0.02
    r2 = 0.004
    reward_class = CenterCTHReward(sim_conf, r1, r2)

    tune_baseline_reward(reward_name, reward_class, r1, r2)

def dist_reward():
    reward_name = f"DistRef"
    reward_class = DistReward()

    tune_baseline_reward(reward_name, reward_class, 1, 0)
    # eval_b_reward(reward_name)
    
def steer_reward():
    r1 = 0.1
    reward_name = f"Steer"
    reward_class = SteeringReward(r1)

    tune_baseline_reward(reward_name, reward_class, r1)

    r1 = 0.01
    reward_name = f"Steer"
    reward_class = SteeringReward(r1)

    tune_baseline_reward(reward_name, reward_class, r1)

    r1 = 0.05
    reward_name = f"Steer"
    reward_class = SteeringReward(r1)

    tune_baseline_reward(reward_name, reward_class, r1)
  
    r1 = 0.02
    reward_name = f"Steer"
    reward_class = SteeringReward(r1)

    tune_baseline_reward(reward_name, reward_class, r1)
    

if __name__ == "__main__":
    cth_reward()

    # dist_reward()
    # steer_reward()

