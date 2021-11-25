from GeneralTestTrain import *
from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import prepare_track_img, ViabilityGenerator
from SupervisorySafetySystem.DiscrimKernel import DiscrimGenerator


def generate_viability_kernel(conf, val, make_picture=False):
    assert conf.kernel_mode == "viab"
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_viab_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    if make_picture:
        kernel.make_picture(False)

    return time_taken

def generate_discriminating_kernel(conf, val, make_picture=False):
    assert conf.kernel_mode == "disc"
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = ViabilityGenerator(img, conf)
    kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_viab_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    if make_picture:
        kernel.make_picture(False)

    return time_taken
    
def generate_standard_kernels():
    conf = load_conf("test_kernel")

    conf.kernel_mode = "viab"
    generate_viability_kernel(conf, "std", True)
    conf.kernel_mode = "disc"
    generate_discriminating_kernel(conf, "std", True)

def kernel_discretization_ndx():
    conf = load_conf("test_kernel")
    env = TrackSim(conf)

    # discretes = [60, 70, 80, 90, 100, 120]
    discretes = [80, 100]
    success_rates = []
    for value in discretes:
        conf.n_dx = value
        kernel_time = generate_viability_kernel(conf, value)

        kernel_name = f"Kernel_viab_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
        success_rates.append(eval_dict['success_rate'])
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "KernelDiscret_ndx" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)

        print(f"Discretizations: {discretes}")
        print(f"Success rates: {success_rates}")
    
def kernel_discretization_time():
    conf = load_conf("test_kernel")
    env = TrackSim(conf)

    # times = [0.09, 0.1, 0.12, 0.15]
    times = [0.1, 0.15]
    success_rates = []
    for value in times:
        conf.kernel_time_step = value
        conf.lookahead_time_step = value * 2    
        kernel_time = generate_viability_kernel(conf, value)

        kernel_name = f"Kernel_viab_{value}_{conf.map_name}.npy"    
        planner = RandomPlanner(f"RandoKernelTest_{value}")

        kernel = TrackKernel(conf, False, kernel_name)
        safety_planner = Supervisor(planner, kernel, conf)

        eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
        success_rates.append(eval_dict['success_rate'])
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "KernelDiscret_time" 
        config_dict['test_number'] = 0
        config_dict['kernel_time'] = kernel_time
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)

        print(f"Times: {times}")
        print(f"Success rates: {success_rates}")

    

def rando_pictures():
    conf = load_conf("test_kernel")
    planner = RandomPlanner("RandoPictures")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    conf.test_n = 5
    render_kernel(env, safety_planner, conf, True)

def rando_results():
    conf = load_conf("test_kernel")
    planner = RandomPlanner("RandoResult")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
    
    config_dict = vars(conf)
    config_dict['EvalName'] = "Safety" 
    config_dict['test_number'] = 0
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)

def straight_test():
    conf = load_conf("test_kernel")
    planner1 = ConstantPlanner("StraightPlanner", 0)
    planner2 = ConstantPlanner("MaxSteerPlanner", 0.4)
    planner3 = ConstantPlanner("MinSteerPlanner", -0.4)

    for planner in [planner1, planner2, planner3]:
        env = TrackSim(conf)
        kernel = TrackKernel(conf, False)
        safety_planner = Supervisor(planner, kernel, conf)

        conf.test_n = 1 # same behaviour
        eval_dict = render_kernel(env, safety_planner, conf, True)
        
        config_dict = vars(conf)
        config_dict['EvalName'] = "Safety" 
        config_dict['test_number'] = 0
        config_dict.update(eval_dict)

        save_conf_dict(config_dict)




if __name__ == "__main__":
    # kernel_discretization_ndx()
    # kernel_discretization_time()

    # generate_standard_kernels()

    rando_pictures()
    rando_results()
    straight_test()

