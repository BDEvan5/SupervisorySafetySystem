
from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.logger import LinkyLogger
from copy import copy
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner

# MAP_NAME = "f1_aut_wide"
MAP_NAME = "columbia_small"

def execute_kernel_run(run_name):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])
    n = runs['n']

    for run in runs['runs']:
        conf = copy(base_config)
        conf['run_name'] = runs['run_name']
        conf['eval_name'] = runs['eval_name']
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
        eval_dict_wo = evaluate_vehicle(env, planner, conf, False)
    
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, False)

        save_dict = vars(conf)
        save_dict['Wo'] = eval_dict_wo
        save_dict['SSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)

def execute_run_tests(run_name):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])

    for run in runs['runs']:
        conf = copy(base_config)
        conf['run_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"KernelSSS_{runs['n']}_{conf.name}"
        link = LinkyLogger(conf, agent_name)
        env = TrackSim(conf, link)

        planner = EndVehicleTest(agent_name, conf)
        eval_dict_wo = evaluate_vehicle(env, planner, conf, False)
    
        planner = EndVehicleTest(agent_name, conf)
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, False)

        save_dict = vars(conf)
        save_dict['Wo'] = eval_dict_wo
        save_dict['SSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)


def execute_gen_run(run_name):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])
    n = runs['n']

    for run in runs['runs']:
        conf = copy(base_config)
        conf['run_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"Rando_{n}_{conf.map_name}_{conf.kernel_mode}"
        link = LinkyLogger(conf, agent_name)
        env = TrackSim(conf, link)

        planner = RandomPlanner(conf, agent_name)
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, False)
    
        save_dict = vars(conf)
        save_dict['SSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)


def execute_performance_kernel_run(n):
    # conf = load_conf("std_test_kernel")
    conf = load_conf("PaperSSS")

    agent_name = f"KernelSSS_{n}"
    link = LinkyLogger(conf, agent_name)
    env = TrackSim(conf, link)

    planner = EndVehicleTrain(agent_name, conf, link)
    safety_planner = LearningSupervisor(planner, conf)
    train_kernel_vehicle(env, safety_planner, conf, show=False)

    planner = EndVehicleTest(agent_name, conf)
    eval_dict_wo = evaluate_vehicle(env, planner, conf, False)

    safety_planner = Supervisor(planner, conf)
    eval_dict_sss = evaluate_vehicle(env, safety_planner, conf, False)

    save_dict = vars(conf)
    save_dict['Wo'] = eval_dict_wo
    save_dict['SSS'] = eval_dict_sss
    save_dict['agent_name'] = agent_name
    save_conf_dict(save_dict)


def train_baseline(n):
    # sim_conf = load_conf("PaperBaseline")
    sim_conf = load_conf("std_test_kernel")
    sim_conf.map_name = MAP_NAME
    agent_name = f"Baseline_{n}"
    link = LinkyLogger(sim_conf, agent_name)
    env = TrackSim(sim_conf, link)

    planner = EndVehicleTrain(agent_name, sim_conf, link)
    train_time, crashes = train_baseline_vehicle(env, planner, sim_conf)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict_wo = evaluate_vehicle(env, planner, sim_conf, False)
    
    save_dict = vars(sim_conf)
    save_dict['train_time'] = train_time
    save_dict['crashes'] = crashes
    save_dict['agent_name'] = agent_name
    save_dict['Wo'] = eval_dict_wo


    save_conf_dict(save_dict)


def run_repeatability():
    for i in range(30, 35):
        execute_performance_kernel_run(i)
        # train_baseline(i)

if __name__ == "__main__":
    # Kernel gen tests
    # execute_gen_run("kernel_gen_run")
    
    # Reward Tests
    # execute_kernel_run("reward_run")
    # execute_run_tests("reward_run")
    
    # Comparision tests
    # execute_performance_kernel_run(21)
    # train_baseline(1)

    run_repeatability()