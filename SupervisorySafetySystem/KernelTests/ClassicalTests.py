from SupervisorySafetySystem.Simulator.TrackSim import TrackSim
from SupervisorySafetySystem.NavAgents.Oracle import Oracle
from SupervisorySafetySystem.NavAgents.FollowTheGap import ForestFGM

from GeneralTestTrain import * 



def test_FGM():
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    # planner = ForestFGM()
    planner = Oracle(sim_conf)
    planner.plan_track(env.env_map)

    sim_conf.test_n = 5

    eval_vehicle(env, planner, sim_conf, True)


def render_picture(n):
    sim_conf = load_conf("track_kernel")
    env = TrackSim(sim_conf)
    planner = ForestFGM()
    # planner = Oracle(sim_conf)
    # planner.plan_track(env.env_map)

    sim_conf.test_n = 4

    render_baseline(env, planner, sim_conf, False)




if __name__ == "__main__":
    # test_FGM()
    render_picture(1)
