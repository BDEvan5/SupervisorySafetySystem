from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavAgents.EndAgentDQN import EndVehicleTrainDQN, EndVehicleTestDQN
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.logger import LinkyLogger


def train_baseline_cth(n, i):
    sim_conf = load_conf("PaperBaseline")
    agent_name = f"BaselineTD3_{n}_{i}"

    link = LinkyLogger(sim_conf, agent_name)
    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)

    train_time, crashes = train_baseline_vehicle(env, planner, sim_conf, False)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperBaseline" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['crashes'] = crashes

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)



def eval_model_sss(n, i):
    # sim_conf = load_conf("PaperSSS")
    sim_conf = load_conf("std_test_kernel")
    agent_name = f"KernelSSS_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = ConstantReward(sim_conf.sss_reward_scale)
    # safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    
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


    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PerfTable" 
    config_dict['test_number'] = n
    # config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

def eval_test():
    n = 4
    i = 2
    sim_conf = load_conf("std_test_kernel")
    # sim_conf = load_conf("BaselineComp")
    agent_name = f"Kernel_ModelSSS_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)
    env = TrackSim(sim_conf, link)
    kernel = TrackKernel(sim_conf, False)
    
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


def eval_test_baseline():
    n = 1
    i = 2
    sim_conf = load_conf("PaperBaseline")
    agent_name = f"BaselineTD3_{n}_{i}"

    
    link = LinkyLogger(sim_conf, agent_name)
    env = TrackSim(sim_conf, link)
    planner = EndVehicleTest(agent_name, sim_conf)
    # planner = EndVehicleTestDQN(agent_name, sim_conf)

    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    # eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Baseline" 
    config_dict['test_number'] = n
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "Retest")




if __name__ == "__main__":
    # train_baseline_cth(1, 3)
    # for i in range(5):
    #     eval_model_sss(5, i)
    eval_model_sss(5, 10)

    # eval_test()
    # eval_test_baseline()
