from SupervisorySafetySystem.KernelTests.GeneralTestTrain import *
from SupervisorySafetySystem.NavAgents.SerialAgentPlanner import SerialVehicleTest, SerialVehicleTrain
from SupervisorySafetySystem.LearningSupervisor import LearningSupervisor
from SupervisorySafetySystem.SupervisorySystem import ForestKernel

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

def train_baseline(VehicleClass, agent_name):
    sim_conf = load_conf("forest_kernel")
    env = ForestSim(sim_conf)
    planner = VehicleClass(agent_name, sim_conf)

    train_vehicle(env, planner, sim_conf)

def test_baseline(VehicleClass, agent_name):
    sim_conf = load_conf("forest_kernel")
    env = ForestSim(sim_conf)
    planner = VehicleClass(agent_name, sim_conf)

    eval_vehicle(env, planner, sim_conf, True)

def train_planner(VehicleClass, agent_name):
    sim_conf = load_conf("forest_kernel")
    env = ForestSim(sim_conf)
    planner = VehicleClass(agent_name, sim_conf)
    kernel = ForestKernel(sim_conf)
    safety_planner = LearningSupervisor(planner, kernel, sim_conf)

    train_kernel_vehicle(env, safety_planner, sim_conf)

def test_planner(VehicleClass, vehicle_name):
    sim_conf = load_conf("forest_kernel")

    env = ForestSim(sim_conf)
    planner = VehicleClass(vehicle_name, sim_conf)
    safety_planner = LearningSupervisor(planner, sim_conf)

    test_kernel_vehicle(env, safety_planner, True, test_n, wait=False)



if __name__ == "__main__":
    # train_baseline(SerialVehicleTrain, baseline_name)
    test_baseline(SerialVehicleTest, baseline_name)


    # train_planner(SerialVehicleTrain, kernel_name)
    # test_planner(SerialVehicleTest, kernel_name)



