
from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.logger import LinkyLogger
from copy import copy

# MAP_NAME = "f1_aut_wide"
MAP_NAME = "columbia_small"

def execute_kernel_run(run_name, n):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])

    for run in runs['runs']:
        conf = copy(base_config)
        conf['run_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"KernelSSS_{n}_{conf.name}"
        link = LinkyLogger(conf, agent_name)
        env = TrackSim(conf, link)

        planner = EndVehicleTrain(agent_name, conf, link)
        safety_planner = LearningSupervisor(planner, conf)
        train_kernel_vehicle(env, safety_planner, conf, show=False)

        planner = EndVehicleTest(agent_name, conf)
        eval_dict_wo = evaluate_vehicle(env, planner, conf, True)
    
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, False)

        save_dict = vars(conf)
        save_dict['ResultsWoSSS'] = eval_dict_wo
        save_dict['ResultsSSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)

def execute_run_tests(run_name, n):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])

    for run in runs['runs']:
        conf = copy(base_config)
        conf['run_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"KernelSSS_{n}_{conf.name}"
        link = LinkyLogger(conf, agent_name)
        env = TrackSim(conf, link)

        planner = EndVehicleTest(agent_name, conf)
        eval_dict_wo = evaluate_vehicle(env, planner, conf, False)
    
        planner = EndVehicleTest(agent_name, conf)
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, True)

        save_dict = vars(conf)
        save_dict['ResultsWoSSS'] = eval_dict_wo
        save_dict['ResultsSSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)



def eval_std_sss():
    sim_conf = load_conf("PaperRewards")
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
    config_dict['EvalName'] = "Reward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['reward'] = 'cth_std'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['reward'] = 'cth_std'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

if __name__ == "__main__":
    execute_kernel_run("reward_run", 1)
    # execute_run_tests("reward_run", 1)
