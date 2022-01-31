from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.logger import LinkyLogger

# MAP_NAME = "f1_aut_wide"
MAP_NAME = "columbia_small"

def train_baseline_cth(n, i):
    sim_conf = load_conf("PaperBaseline")
    sim_conf.map_name = MAP_NAME
    agent_name = f"Baseline_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)

    train_time, crashes = train_baseline_vehicle(env, planner, sim_conf)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Repeatability" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def eval_model_sss(n, i):
    sim_conf = load_conf("std_test_kernel")
    # sim_conf = load_conf("PaperSSS")
    sim_conf.map_name = MAP_NAME
    agent_name = f"KernelSSS_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = ConstantReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Repeatability" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Repeatability"
    config_dict['test_number'] = n
    # config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")


def eval_model_sss_mag(n, i):
    sim_conf = load_conf("std_test_kernel")
    # sim_conf = load_conf("PaperSSS")
    sim_conf.map_name = MAP_NAME
    agent_name = f"KernelSSS_mag_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Repeatability" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Repeatability"
    config_dict['test_number'] = n
    # config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")



if __name__ == "__main__":
    eval_model_sss(2, 0)
    # for i in range(3):
        # train_baseline_cth(2, i)
        # eval_model_sss(2, i)
        # eval_model_sss_mag(2, i)

