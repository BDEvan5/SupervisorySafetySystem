from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *

from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator

import numpy as np
from matplotlib import pyplot as plt


def generate_single_kernel():
    conf = load_conf("track_kernel")

    value = 0

    generate_kernel(conf, value)

def generate_kernel(conf, val):
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"TestKernel_{conf.track_kernel_path}_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    # kernel.view_build(False)
    kernel.make_picture(True)

    return time_taken
    
def rand_kernel_safety_discretization():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    # discretes = [60, 70, 80, 90, 100, 120]
    # discretes = [70, 90]
    discretes = [60, 70, 85, 90]
    success_rates = []
    for value in discretes:
        conf.n_dx = value
        kernel_time = generate_kernel(conf, value)

        kernel_name = f"TestKernel_{conf.track_kernel_path}_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = eval_vehicle(env, safety_planner, conf, False)
        success_rates.append(eval_dict['success_rate'])
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "PaperTest" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)

        print(f"Discretizations: {discretes}")
        print(f"Success rates: {success_rates}")


def run_single_timestep():
    conf = load_conf("track_kernel")
    env = TrackSim(conf)

    value = 0.15
    conf.n_dx = 80

    # conf.time_step = value
    conf.kernel_time_step = value
    conf.lookahead_time_step = value * 2
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

    times = [0.09, 0.1, 0.12, 0.15]
    success_rates = []

    for value in times:
        conf.kernel_time_step = value
        conf.kernel_time_step = value
        conf.lookahead_time_step = value * 2    
        kernel_time = generate_kernel(conf, value)

        kernel_name = f"TestKernel_{conf.track_kernel_path}_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = eval_vehicle(env, safety_planner, conf, False)
        success_rates.append(eval_dict['success_rate'])
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "PaperTest" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)


        print(f"Times: {times}")
        print(f"Success rates: {success_rates}")

if __name__ == "__main__":
    generate_single_kernel()
    # rand_kernel_safety_discretization()
    # rand_kernel_safety()
    # rand_kernel_safety_timestep()
    # run_single_timestep()

