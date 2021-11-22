from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
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

def run_single_timestep():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    value = 0.1
    conf.n_dx = 80

    conf.time_step = value
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


def rand_kernel_safety_timestep():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    for value in [0.09, 0.1, 0.11]:
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

if __name__ == "__main__":
    # generate_kernels()
    # rand_kernel_safety()
    # rand_kernel_safety_timestep()
    run_single_timestep()

