from SupervisorySafetySystem.KernelTests.GeneralTestTrain import train_vehicle, test_single_vehicle
from SupervisorySafetySystem.NavAgents.SerialAgentPlanner import SerialVehicleTest, SerialVehicleTrain

# from SupervisorySafetySystem.KernelTests.ConstructingKernel import DiscriminatingImgKernel, Kernel, SafetyPlannerPP, KernelSim
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim

import yaml
from argparse import Namespace

test_n = 100
run_n = 2
baseline_name = f"std_sap_baseline_{run_n}"

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



def train_planner(agent_name):
    sim_conf = load_conf("kernel_config")
    env = ForestSim(sim_conf)
    vehicle = SerialVehicleTrain(agent_name, sim_conf)

    train_vehicle(env, vehicle, sim_conf)

def test_planner(vehicle_name):
    sim_conf = load_conf("kernel_config")
    env = ForestSim(sim_conf)
    vehicle = SerialVehicleTest(vehicle_name, sim_conf)

    test_single_vehicle(env, vehicle, True, test_n, wait=False)



if __name__ == "__main__":
    # train_planner(baseline_name)
    test_planner(baseline_name)



