# from toy_auto_race.NavAgents.FollowTheGap import ForestFGM
from toy_auto_race.Utils import LibFunctions as lib
from SupervisorySafetySystem.Simulator.ForestSim import ForestSim
# from toy_auto_race.NavAgents.SafetyCar import SafetyCar
# from toy_auto_race.NavAgents.SafetyEnvelope import SafetyCar
# from toy_auto_race.NavAgents.SuperSafetySystem import SafetyCar
# from toy_auto_race.NavAgents.DynamicWindow import SafetyCar
# from toy_auto_race.NavAgents.TwoDWA import SafetyCar
# from SupervisorySafetySystem.SafetySys.TwoDWAxy import SafetyCar
from SupervisorySafetySystem.SafetySys.VelObs import SafetyCar
from SupervisorySafetySystem.NavAgents.RandoCar import RandoCar

from SupervisorySafetySystem.Histories import HistoryManager

from TrainTest import *


def test_safety_system():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = SafetyCar(sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)
    # test_single_vehicle(env, vehicle, False, 100, wait=False)


def test_safety_system_random():
    sim_conf = lib.load_conf("std_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = RandoCar(sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)
    # test_single_vehicle(env, vehicle, False, 100, wait=False)


def test_forest_system():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = ForestFGM()

    test_single_vehicle(env, vehicle, True, 100)

def run_data_bag():
    his_manager = HistoryManager()
    his_manager.open_history(90)
    his_manager.step_run()

if __name__ == "__main__":
    test_safety_system()
    # test_safety_system_random()
    # test_forest_system()

    # run_data_bag()


