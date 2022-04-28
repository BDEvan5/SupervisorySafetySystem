from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM
from SupervisorySafetySystem.NavAgents.SimplePlanners import PurePursuit

from GeneralTestTrain import * 
from SupervisorySafetySystem.logger import LinkyLogger


def test_FGM():
    sim_conf = load_conf("PaperClassical")
    planner = ForestFGM(sim_conf)
    link = LinkyLogger(sim_conf, planner.name)
    env = TrackSim(sim_conf, link)

    sim_conf.test_n = 5

    evaluate_vehicle(env, planner, sim_conf, True)
    # render_baseline(env, planner, sim_conf, False)


def test_oracle():
    conf = load_conf("PaperClassical")
    planner = PurePursuit(conf)
    link = LinkyLogger(conf, planner.name)
    env = TrackSim(conf, link)

    conf.test_n = 1

    evaluate_vehicle(env, planner, conf, True)
    # render_baseline(env, planner, conf, False)



if __name__ == "__main__":
    test_FGM()
    # test_oracle()

