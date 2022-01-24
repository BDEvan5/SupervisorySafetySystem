import numpy as np 
from SupervisorySafetySystem.NavUtils.TD3 import TD3
from SupervisorySafetySystem.NavUtils.HistoryStructs import TrainHistory
from SupervisorySafetySystem.NavUtils.speed_utils import calculate_speed
from SupervisorySafetySystem.NavUtils.RewardFunctions import *
import torch
from SupervisorySafetySystem.NavUtils.DQN import DQN
from SupervisorySafetySystem.Modes import Modes
from SupervisorySafetySystem.NavAgents.EndAgent import EndBase

class EndVehicleTrainDQN(EndBase):
    def __init__(self, agent_name, sim_conf, load=False):
        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.vehicle_path + agent_name
        state_space = 2 + self.n_beams
        self.m = Modes(sim_conf)
        self.agent = DQN(state_space, self.m.n_modes, agent_name)
        self.agent.try_load(load, sim_conf.h_size, self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(agent_name, sim_conf, load)

        # self.calculate_reward = DistReward() 
        # self.calculate_reward = CthReward(0.04, 0.004) 
        # self.calculate_reward = SteeringReward(0.01) 
        # self.calculate_reward = None
        self.calculate_reward = RefCTHReward(sim_conf) 

    def plan_act(self, obs, add_mem_entry=True):
        nn_obs = self.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action

        self.nn_state = nn_obs

        # steering_angle = nn_action[0] * self.max_steer
        # speed = calculate_speed(steering_angle)
        # self.action = np.array([steering_angle, speed])
        self.action = self.m.qs[nn_action]

        return self.action

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.calculate_reward(self.state, s_prime)

            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.calculate_reward(self.state, s_prime)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
            self.agent.save(self.path)
            print(f"Exp: {self.agent.exploration_rate}")
        self.state = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def fake_done_entry(self, s_prime):
        """
        To be called when the supervisor intervenes
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.calculate_reward(self.state, s_prime)

        self.t_his.add_step_data(reward)
        self.state = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def fake_done(self):
        """
        To be called when ep is done.
        """
        self.t_his.lap_done(False)
        self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
            self.agent.save(self.path)




class EndVehicleTestDQN(EndBase):
    def __init__(self, agent_name, sim_conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.vehicle_path + agent_name
        self.model = torch.load(self.path + '/' + agent_name + "_model.pth")
        self.m = Modes(sim_conf)
        # self.n_beams = 10

        print(f"Agent loaded: {agent_name}")

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        out = self.model.forward(nn_obs)
        nn_action = out.argmax().item()

        action = self.m.qs[nn_action]

        return action
