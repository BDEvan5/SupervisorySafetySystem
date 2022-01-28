
from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
from SupervisorySafetySystem.logger import LinkyLogger

# MAP_NAME = "f1_aut_wide"
MAP_NAME = "columbia_small"

#1 std learning
def eval_std_sss(n, i):
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

# 2zero sss
def eval_zero_sss(n, i):
    sim_conf = load_conf("PaperRewards")
    # sim_conf = load_conf("PaperSSS")
    sim_conf.map_name = MAP_NAME
    agent_name = f"KernelSSS_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = ZeroReward()
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Zero"
    config_dict['reward'] = 'cth_std'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Zero"
    config_dict['reward'] = 'cth_std'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

#3 std baseline reward
def eval_ct0_sss(n, i):
    sim_conf = load_conf("PaperRewards")
    sim_conf.rk = 0
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
    config_dict['reward'] = 'cth_0'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['reward'] = 'cth_0'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

# 4std with rk = 0.1
def eval_ct01_sss(n, i):
    sim_conf = load_conf("PaperRewards")
    sim_conf.rk = 0.1
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
    config_dict['reward'] = 'cth_01'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['reward'] = 'cth_01'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

# 5no cth with rk=0.1
def eval_t01_sss(n, i):
    sim_conf = load_conf("PaperRewards")
    sim_conf.rk = 0.1
    sim_conf.r1 = 0
    sim_conf.r2 = 0
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
    config_dict['reward'] = 't_01'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['reward'] = 't_01'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")

#6 zero sss with pure time
def eval_t01zero_sss(n, i):
    sim_conf = load_conf("PaperRewards")
    sim_conf.rk = 0.1
    sim_conf.r1 = 0
    sim_conf.r2 = 0
    # sim_conf = load_conf("PaperSSS")
    sim_conf.map_name = MAP_NAME
    agent_name = f"KernelSSS_{n}_{i}"
    link = LinkyLogger(sim_conf, agent_name)

    env = TrackSim(sim_conf, link)
    planner = EndVehicleTrain(agent_name, sim_conf, link)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = ZeroReward()
    # safety_planner.calculate_reward = ConstantReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "zero"
    config_dict['reward'] = 't_01'

    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

    planner = EndVehicleTest(agent_name, sim_conf)
    eval_dict = evaluate_vehicle(env, planner, sim_conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "Reward"
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Zero"
    config_dict['reward'] = 't_01'
    config_dict['vehicle'] = "KernelWoSSS"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "WithoutSSS")





if __name__ == "__main__":
    # eval_std_sss(3, 1)
    # eval_zero_sss(3, 2)
    # eval_ct0_sss(3, 3)
    # eval_ct01_sss(3, 4)
    # eval_t01_sss(3, 5)
    eval_t01zero_sss(3, 6)
