from GeneralTestTrain import *
from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.SupervisorySystem import Supervisor, TrackKernel, LearningSupervisor
from SupervisorySafetySystem.NavAgents.SimplePlanners import RandomPlanner, ConstantPlanner
from SupervisorySafetySystem.KernelRewards import *

from SupervisorySafetySystem.KernelGenerator import build_track_kernel
from SupervisorySafetySystem.DynamicsBuilder import build_dynamics_table
from SupervisorySafetySystem.logger import LinkyLogger
    
def generate_standard_kernels():
    conf = load_conf("PaperKernelGen")

    conf.kernel_mode = "viab"
    build_dynamics_table(conf)
    for name in ["columbia_small", "porto", "f1_aut_wide"]:
    # for name in ["columbia_small"]:
        conf.map_name = name
        build_track_kernel(conf)

    # conf.kernel_mode = "disc"
    # # build_dynamics_table(conf)
    # for name in ["columbia_small", "porto", "f1_aut_wide"]:
    # # for name in ["columbia_small"]:
    #     conf.map_name = name
    #     build_track_kernel(conf)


def kernel_discretization_ndx():
    conf = load_conf("SafetyTests")
    conf.EvalName = "KernelDiscret_ndx" 
    conf.test_n = 100
    env = TrackSim(conf)

    discretes = [60, 70, 80, 90, 100, 120]
    # discretes = [80, 100]
    success_rates = []
    for value in discretes:
        conf.n_dx = value

        for mode in ["viab", "disc"]:
            conf.kernel_mode = mode
            if mode == "viab":
                kernel_time, kernel_filled = generate_viability_kernel(conf, value)
            elif mode == "disc":
                kernel_time, kernel_filled = generate_discriminating_kernel(conf, value)

            kernel_name = f"Kernel_{conf.kernel_mode}_{value}_{conf.map_name}.npy"    
            planner = RandomPlanner(f"RandoKernelTest_{conf.kernel_mode}_{value}")

            kernel = TrackKernel(conf, False, kernel_name)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
            success_rates.append(eval_dict['success_rate'])
            
            config_dict = vars(conf)
            config_dict['kernel_time'] = kernel_time
            config_dict['kernel_filled'] = kernel_filled
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)

    print(f"Discretizations: {discretes}")
    print(f"Success rates: {success_rates}")


def kernel_discretization_time():
    conf = load_conf("SafetyTests")
    conf.EvalName = "KernelDiscret_time" 
    conf.test_n = 100
    env = TrackSim(conf)

   # times = [0.09, 0.1, 0.12, 0.15]
    times = [0.1, 0.15]
    success_rates = []
    for value in times:
        conf.kernel_time_step = value
        conf.lookahead_time_step = value * 2  

        conf.kernel_mode = "viab"
        kernel_time = generate_viability_kernel(conf, value)
        conf.kernel_mode = "disc"
        kernel_time = generate_discriminating_kernel(conf, value)

        for mode in ["viab", "disc"]:
            conf.kernel_mode = mode
            kernel_name = f"Kernel_{conf.kernel_mode}_{value}_{conf.map_name}.npy"    
            planner = RandomPlanner(f"RandoKernelTest_{conf.kernel_mode}_{value}")

            kernel = TrackKernel(conf, False, kernel_name)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
            success_rates.append(eval_dict['success_rate'])
            
            config_dict = vars(conf)
            config_dict['kernel_time'] = kernel_time
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)

    print(f"Discretizations: {times}")
    print(f"Success rates: {success_rates}")
    
    

def rando_pictures():
    conf = load_conf("SafetyTests")
    planner = RandomPlanner("RandoPictures")

    env = TrackSim(conf)
    kernel = TrackKernel(conf, False)
    safety_planner = Supervisor(planner, kernel, conf)

    conf.test_n = 5
    render_kernel(env, safety_planner, conf, True)

# def rando_results(n):
#     conf = load_conf("PaperSafety")
#     conf.kernel_mode = "disc"
#     conf.vehicle = "random"
#     conf.test_n = 10

#     agent_name = f"RandoResult_{conf.kernel_mode}_{n}"
#     planner = RandomPlanner(conf, agent_name)
#     link = LinkyLogger(conf, agent_name)
#     env = TrackSim(conf, link)
#     # for mode in ["viab", "disc"]:
#     for mode in ["disc"]:
#         conf.kernel_mode = mode

#         kernel = TrackKernel(conf, False)
#         # kernel.print_kernel_area()
#         safety_planner = Supervisor(planner, kernel, conf)

#         eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
        
#         config_dict = vars(conf)
#         config_dict['test_number'] = n
#         config_dict.update(eval_dict)

#         save_conf_dict(config_dict)

def rando_results(n):
    conf = load_conf("PaperKernelGen")
    conf.vehicle = "random"
    conf.test_n = 10

    for map_name in ["porto", "columbia_small", "f1_aut_wide"]:
        conf.map_name = map_name
        for mode in ["viab", "disc"]:
            conf.kernel_mode = mode 
            agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
            planner = RandomPlanner(conf, agent_name)
            link = LinkyLogger(conf, agent_name)
            env = TrackSim(conf, link)

            kernel = TrackKernel(conf, False)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = evaluate_vehicle(env, safety_planner, conf, False)
            
            config_dict = vars(conf)
            config_dict['test_number'] = n
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)

def run_constant_tests(n):
    conf = load_conf("SafetyTests")
    conf.vehicle = "constant"

    env = TrackSim(conf)
    for mode in ["viab", "disc"]:
        conf.kernel_mode = mode
        kernel = TrackKernel(conf, False)
        for value in [0, -0.4, 0.4]:
            conf.constant_value = value
            planner = ConstantPlanner(f"ConstPlanner_{value}_{conf.kernel_mode}_{n}", conf.constant_value)
            safety_planner = Supervisor(planner, kernel, conf)

            eval_dict = render_kernel(env, safety_planner, conf, True)
            
            config_dict = vars(conf)
            config_dict['test_number'] = n
            config_dict.update(eval_dict)

            save_conf_dict(config_dict)






if __name__ == "__main__":
    # kernel_discretization_ndx()
    # kernel_discretization_time()

    # generate_standard_kernels()

    # rando_pictures()
    rando_results(1)
    # run_constant_tests(1)


