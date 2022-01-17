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
    filled = kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_viab_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    if make_picture:
        kernel.make_picture(False)

    return time_taken, filled

def generate_discriminating_kernel(conf, val, make_picture=False):
    assert conf.kernel_mode == "disc"
    start_time = time.time()
    img = prepare_track_img(conf) 
    kernel = DiscrimGenerator(img, conf)
    filled = kernel.calculate_kernel(50)
    kernel.save_kernel(f"Kernel_disc_{val}_{conf.map_name}")
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
    kernel.view_build(False)
    if make_picture:
        kernel.make_picture(False)

    return time_taken, filled
    
def generate_standard_kernels():
    conf = load_conf("std_test_kernel")

    # conf.kernel_mode = "viab"
    # generate_viability_kernel(conf, "std", True)
    conf.kernel_mode = "disc"
    generate_discriminating_kernel(conf, "std", True)







if __name__ == "__main__":

    generate_standard_kernels()

    # rando_pictures()
    # rando_results(1)
    # run_constant_tests(1)


