import numpy as np


class LinkyLogger:
    def __init__(self, conf, agent_name):
        self.path = "Data/Logs"  
        # self.path = conf.vehicle_path + agent_name 

        self.env_log = "/env_log.txt"
        self.agent_log = "/agent_log.txt"

        with open(self.path + self.env_log, "w") as f:
            f.write(agent_name + "\n")

        with open(self.path + self.agent_log, "w") as f:
            f.write(agent_name + "\n")

    def write_env_log(self, data):
        with open(self.path + self.env_log, "a") as f:
            f.write(data)

    def write_agent_log(self, data):
        with open(self.path + self.agent_log, "a") as f:
            f.write(data)

