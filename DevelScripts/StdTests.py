from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *



def train_baseline_cth(n, i):
    sim_conf = load_conf("std_test_kernel")
    reward = "cthRef"
    agent_name = f"Baseline_{reward}_{i}_{n}"

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)

    train_time, crashes = train_baseline_vehicle(env, planner, sim_conf)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "BaselineComp" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['reward'] = reward
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def eval_model_sss(n, i):
    sim_conf = load_conf("std_test_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_ModelSSS_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "BaselineComp" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['learning'] = "Continuous"
    config_dict['kernel_reward'] = "Magnitude"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_test():
    n = 1
    i = 1
    sim_conf = load_conf("BaselineComp")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_ModelSSS_{i}_{n}"
    kernel = TrackKernel(sim_conf)
    
    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "BaselineComp" 
    config_dict['test_number'] = n
    config_dict['learning'] = "Continuous"
    config_dict['kernel_reward'] = "Magnitude"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "Retest")




if __name__ == "__main__":
    # train_baseline_cth(1, 1)
    eval_model_sss(1, 2)

    # eval_test()
