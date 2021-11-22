from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.NavAgents.EndAgent import EndVehicleTrain, EndVehicleTest
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator

import numpy as np
from matplotlib import pyplot as plt


def generate_kernels():
    conf = load_conf("track_kernel")

    # for value in [50, 80, 120, 160]:
    #     conf.n_dx = value
    #     generate_kernel(conf, value)

    # value = 200
    # conf.n_dx = value

    value = 0.1
    conf.kernel_time_step

    generate_kernel(conf, value)

def generate_kernel(conf, val):
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"TestKernel_{conf.track_kernel_path}_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)

    return time_taken
    
def rand_kernel_safety():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    for value in [50, 80, 100, 150, 200]:
        conf.n_dx = value
        kernel_time = generate_kernel(conf, value)

        kernel_name = f"TestKernel_{conf.track_kernel_path}_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = eval_vehicle(env, safety_planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "PaperTest" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)


def rand_kernel_safety_timestep():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    for value in [0.08, 0.1, 0.12]:
        conf.kernel_time_step = value
        kernel_time = generate_kernel(conf, value)

        kernel_name = f"TestKernel_{conf.track_kernel_path}_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = eval_vehicle(env, safety_planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "PaperTest" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)

def rando_pictures():
    conf = load_conf("track_kernel")
    planner = RandomPlanner("RandoPictures")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    conf.test_n = 5
    eval_dict = eval_kernel(env, safety_planner, conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = 0
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def rando_results():
    conf = load_conf("track_kernel")
    planner = RandomPlanner("RandoResult")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    eval_dict = eval_vehicle(env, safety_planner, conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = 0
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def straight_test():
    conf = load_conf("track_kernel")
    # planner = ConstantPlanner("StraightPlanner", 0)
    # planner = ConstantPlanner("MaxSteerPlanner", 0.4)
    planner = ConstantPlanner("MinSteerPlanner", -0.4)

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    conf.test_n = 1
    eval_dict = eval_kernel(env, safety_planner, conf, True)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = 0
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def eval_magnitude_reward(sss_reward_scale, n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Mag_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = eval_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Magnitude"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def eval_constant_reward(sss_reward_scale, n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    agent_name = f"Kernel_Const_{sss_reward_scale}_{n}"
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(sss_reward_scale)
    
    train_time = train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

    planner = EndVehicleTest(agent_name, sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    eval_dict = eval_vehicle(env, safety_planner, sim_conf, False)
    
    config_dict = vars(sim_conf)
    config_dict['EvalName'] = "PaperTest" 
    config_dict['test_number'] = n
    config_dict['train_time'] = train_time
    config_dict['kernel_reward'] = "Constant"
    config_dict['sss_reward_scale'] = sss_reward_scale
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def run_reward_tests():
    n = 1
    for n in range(5):
        eval_constant_reward(0, n)
        eval_constant_reward(0.2, n)
        eval_constant_reward(0.5, n)
        eval_constant_reward(0.7, n)
        eval_constant_reward(1, n)

        eval_magnitude_reward(0.2, n)
        eval_magnitude_reward(0.5, n)
        eval_magnitude_reward(0.7, n)
        eval_magnitude_reward(1, n)


def learning_comparision_sss():
    sim_conf = load_conf("track_kernel")
    n = 1
    eval_name = f"LearningComparison_SSS_{n}"
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    agent_name = f"Kernel_Const_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Const_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)    

    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Mag_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)   

    agent_name = f"Kernel_Mag_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)


    test.run_free_eval(env, 100, wait=False)

def learning_comparision_pure():
    sim_conf = load_conf("track_kernel")
    n = 1
    eval_name = f"LearningComparison_pure_{n}"
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    agent_name = f"Kernel_Const_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Const_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)    

    agent_name = f"Kernel_Const_{0}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Mag_{0.5}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)   

    agent_name = f"Kernel_Mag_{1}_{n}"
    planner = EndVehicleTest(agent_name, sim_conf)
    test.add_vehicle(planner)


    test.run_free_eval(env, 100, wait=False)





def train_baseline(agent_name):
    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)

    train_vehicle(env, planner, sim_conf)

def test_baseline(agent_name):
    env = TrackSim(sim_conf)
    planner = EndVehicleTest(agent_name, sim_conf)

    eval_vehicle(env, planner, sim_conf, True)

def test_FGM():
    env = TrackSim(sim_conf)
    planner = ForestFGM()
    # planner = Oracle(sim_conf)

    eval_vehicle(env, planner, sim_conf, True)

def train_learning_kerns():

    env = TrackSim(sim_conf)
    planner = EndVehicleTrain(agent_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)
    safety_planner.calculate_reward = MagnitudeReward(0.5)
    safety_planner.calculate_reward = MagnitudeReward(1)
    safety_planner.calculate_reward = ConstantReward(0.5)
    safety_planner.calculate_reward = ConstantReward(1)
    safety_planner.calculate_reward = ZeroReward()

    train_kernel_vehicle(env, safety_planner, sim_conf, show=False)

def test_kernel_sss(vehicle_name):
    env = TrackSim(sim_conf)
    planner = EndVehicleTest(vehicle_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)

    test_kernel_vehicle(env, safety_planner, False, test_n, wait=False)
    # test_kernel_vehicle(env, safety_planner, True, test_n, wait=False)

def test_kernel_pure(vehicle_name):
    env = TrackSim(sim_conf)
    planner = EndVehicleTest(vehicle_name, sim_conf)
    # kernel = TrackKernel(sim_conf)
    # safety_planner = Supervisor(planner, kernel, sim_conf)

    # eval_vehicle(env, planner, sim_conf, True)
    eval_vehicle(env, planner, sim_conf, False)
    # test_kernel_vehicle(env, planner, True, test_n, wait=False)

def baseline_vs_kernel(baseline_name, kernel_name):
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    baseline = EndVehicleTest(baseline_name, sim_conf)
    test.add_vehicle(baseline)

    planner = EndVehicleTest(kernel_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)

    test.run_free_eval(env, test_n, wait=False)


def full_comparison(baseline_name, kernel_name):
    test = TestVehicles(sim_conf, eval_name)
    env = TrackSim(sim_conf)
    
    baseline = EndVehicleTest(baseline_name, sim_conf)
    test.add_vehicle(baseline)

    planner = EndVehicleTest(kernel_name, sim_conf)
    kernel = TrackKernel(sim_conf)
    safety_planner = Supervisor(planner, kernel, sim_conf)
    test.add_vehicle(safety_planner)

    vehicle = ForestFGM()
    test.add_vehicle(vehicle)

    vehicle = Oracle(sim_conf)
    test.add_vehicle(vehicle)

    test.run_free_eval(env, test_n, wait=False)




if __name__ == "__main__":
    # train_baseline(baseline_name)
    # test_baseline(baseline_name)
    # test_FGM()

    # train_kenel(kernel_name)
    # test_kernel_sss(kernel_name)
    # test_kernel_sss(baseline_name)
    # test_kernel_pure(kernel_name)

    # baseline_vs_kernel(baseline_name, kernel_name)
    # full_comparison(baseline_name, kernel_name)



    # generate_kernels()
    # rand_kernel_safety()
    rand_kernel_safety_timestep()

    # rando_results()
    # rando_pictures()
    # straight_test()

    # run_reward_tests()
    # learning_comparision_sss()
    # learning_comparision_pure()


