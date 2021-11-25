from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM

from GeneralTestTrain import * 



def test_FGM():
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    planner = ForestFGM(sim_conf)

    sim_conf.test_n = 5

    evaluate_vehicle(env, planner, sim_conf, True)
    render_baseline(env, planner, sim_conf, False)


def test_oracle():
    sim_conf = load_conf("test_kernel")
    env = TrackSim(sim_conf)
    planner = Oracle(sim_conf)
    planner.plan_track(env.env_map)

    sim_conf.test_n = 5

    evaluate_vehicle(env, planner, sim_conf, True)
    render_baseline(env, planner, sim_conf, False)



if __name__ == "__main__":
    test_FGM()
    test_oracle()

