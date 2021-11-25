from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *


# Kernel Reward Tests
def eval_magnitude_reward(sss_reward_scale, n):
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Mag_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_episodic(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "KernelReward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_constant_reward(sss_reward_scale, n):
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Const_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_episodic(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "KernelReward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def kernel_reward_tests():
    # n = 1
    for n in range(1):
        eval_constant_reward(0, n)
        # eval_constant_reward(0.2, n)
        # eval_constant_reward(0.5, n)
        # eval_constant_reward(0.7, n)
        eval_constant_reward(1, n)

        # eval_magnitude_reward(0.2, n)
        # eval_magnitude_reward(0.5, n)
        # eval_magnitude_reward(0.7, n)
        eval_magnitude_reward(1, n)


def render_picture(n):
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    sim_conf.test_n = 4

    agent_name = f"Kernel_Mag_NoTrain_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    render_kernel(env, safety_planner, sim_conf, False)

    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    render_kernel(env, safety_planner, sim_conf, False)

    agent_name = f"Kernel_Mag_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf, False)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    render_kernel(env, safety_planner, sim_conf, False)

def eval_continuous(n):
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Continuous_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RefCTHRewardContinuous(sim_conf, 0.04, 0.004) 
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "LearningMode" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['learning'] = "Continuous"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_episodic(n):
    sss_reward_scale = 1
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Episodic_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    planner.calculate_reward = RefCTHReward(sim_conf, 0.04, 0.004) 
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_episodic(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "LearningMode" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['learning'] = "Episodic"
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


if __name__ == "__main__":
    # kernel_reward_tests()
    # render_picture()

    eval_continuous(1)
    eval_episodic(1)
