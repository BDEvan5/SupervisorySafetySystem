

from LearningLocalPlanning.NavAgents.AgentNav import NavTrainVehicle
from LearningLocalPlanning.NavAgents.AgentMod import ModVehicleTrain

from LearningLocalPlanning.Simulator.ForestSim import ForestSim
import yaml   
from argparse import Namespace



def train_vehicle(env, vehicle, steps):
    done = False
    state = env.reset()

    print(f"Starting Training: {vehicle.name}")
    for n in range(steps):
        a = vehicle.plan_act(state)
        s_prime, r, done, _ = env.step_plan(a)

        state = s_prime
        vehicle.agent.train(2)
        
        # env.render(False)
        
        if done:
            vehicle.done_entry(s_prime)
            # vehicle.show_vehicle_history()
            # env.history.show_history()
            # env.render(wait=False, name=vehicle.name)

            vehicle.reset_lap()
            state = env.reset()

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()

    print(f"Finished Training: {vehicle.name}")



map_name = "forest2"
n = 3
nav_name = f"Navforest_{n}"
mod_name = f"ModForest_{n}"
repeat_name = f"RepeatTest_{n}"
eval_name = f"CompareTest_{n}"
train_n = 1000

def load_conf(path, fname):
    full_path = path + 'config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



"""
Training Functions
"""
def train_nav():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = NavTrainVehicle(nav_name, sim_conf, h_size=200)

    train_vehicle(env, vehicle, train_n)


def train_mod():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)
    vehicle = ModVehicleTrain(mod_name, map_name, sim_conf, load=False, h_size=200)

    train_vehicle(env, vehicle, train_n)



def train_repeatability():
    sim_conf = load_conf("", "std_config")
    env = ForestSim(map_name, sim_conf)

    for i in range(10):
        train_name = f"ModRepeat_forest_{i}"

        vehicle = ModVehicleTrain(train_name, map_name, sim_conf, load=False)

        train_vehicle(env, vehicle, train_n)


if __name__ == "__main__":
    train_mod()
    train_nav()
    train_repeatability()
