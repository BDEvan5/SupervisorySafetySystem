from SupervisorySafetySystem.KernelTests.GeneralTestTrain import train_vehicle, test_single_vehicle
from SupervisorySafetySystem.NavAgents.SerialAgentPlanner import SerialVehicleTest, SerialVehicleTrain
from SupervisorySafetySystem.KernelTests.AgentKernel import AgentKernelTrain, AgentKernelTest

from SupervisorySafetySystem.KernelTests.ConstructingKernel import DiscriminatingImgKernel, Kernel, SafetyPlannerPP
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim

import yaml
from argparse import Namespace
import numpy as np

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"
kernel_name = f"kernel_sap_{run_n}"

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



def train_planner(VehicleClass, agent_name):
    sim_conf = load_conf("kernel_config")
    env = ForestSim(sim_conf)
    vehicle = VehicleClass(agent_name, sim_conf)

    train_vehicle(env, vehicle, sim_conf)

def test_planner(VehicleClass, vehicle_name):
    sim_conf = load_conf("kernel_config")
    env = ForestSim(sim_conf)
    vehicle = VehicleClass(vehicle_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False)


def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size / conf.resolution)
    obs_size = int(conf.obs_size / conf.resolution)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = DiscriminatingImgKernel(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def constructy_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) / conf.resolution , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = DiscriminatingImgKernel(img, conf)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")





if __name__ == "__main__":
    # train_planner(SerialVehicleTrain, baseline_name)
    # test_planner(SerialVehicleTest, baseline_name)


    # train_planner(AgentKernelTrain, kernel_name)
    test_planner(AgentKernelTest, kernel_name)



