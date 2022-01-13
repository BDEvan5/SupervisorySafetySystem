from GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.KernelRewards import *
from SupervisorySafetySystem.NavUtils.RewardFunctions import *


# Kernel Reward Tests
def eval_magnitude_reward(sss_reward_scale, i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Mag_{sss_reward_scale}_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    # safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    # eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "KernelReward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_constant_reward(sss_reward_scale, i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Const_{sss_reward_scale}_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_continuous(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    # safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    # eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "KernelReward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def kernel_reward_tests(n):
    # n = 1
    for i in range(2, 5):
        eval_constant_reward(0, i, n)
        eval_constant_reward(0.2, i, n)
        # eval_constant_reward(0.5, i, n)
        # eval_constant_reward(0.7, i, n)
        eval_constant_reward(1, i, n)

        eval_magnitude_reward(0.2, i, n)
        # eval_magnitude_reward(0.5, i, n)
        # eval_magnitude_reward(0.7, i, n)
        eval_magnitude_reward(1, i, n)
        # eval_magnitude_reward(2, i, n)
        # eval_magnitude_reward(4, i, n)

def test_vehicle():
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)

    sss_reward_scale = 1
    i = 6
    n = 1

    agent_name = f"Kernel_Const_{sss_reward_scale}_{i}_{n}"
    # agent_name = f"Kernel_Mag_{sss_reward_scale}_{i}_{n}"
    
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_dict = evaluate_vehicle(env, planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "KernelReward" 
    config_dict['test_number'] = n
    config_dict['train_time'] = 0
    config_dict['kernel_reward'] = "Constant"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict, "PureTest")

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

def eval_continuous_mag(i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Continuous_mag_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
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
    config_dict['learning_mode'] = "ContinuousMag"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_episodic_mag(i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Episodic_mag_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    
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
    config_dict['learning_mode'] = "EpisodicMag"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_continuous_const(i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Continuous_const_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = ConstantReward(sim_conf.sss_reward_scale)
    
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
    config_dict['learning_mode'] = "ContinuousConst"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_episodic_const(i, n):
    sim_conf = load_conf("FormulationTests")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Episodic_const_{i}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sim_conf.sss_reward_scale)
    safety_planner.calculate_reward = ConstantReward(sim_conf.sss_reward_scale)
    
    train_time = train_kernel_episodic(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = evaluate_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "LearningMode" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['learning'] = "Episodic"
    config_dict['kernel_reward'] = "Constant"
    config_dict['learning_mode'] = "EpisodicConst"
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def learning_mode_tests(n):
    for i in range(10):
        eval_continuous_mag(i, n)
        eval_episodic_mag(i, n)
        eval_continuous_const(i, n)
        eval_episodic_const(i, n)


if __name__ == "__main__":
    kernel_reward_tests(2)
    # render_picture()

    # eval_continuous(3)
    # eval_episodic(1)

    # learning_mode_tests(6)
    # test_vehicle()
