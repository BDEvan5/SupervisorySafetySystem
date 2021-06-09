import numpy as np
from SupervisorySafetySystem.SafetySys.ExportSSS import run_safety_check



class SafetyHistory:
    def __init__(self):
        self.states = []
        
    def add_state(self, obs, pp_action, s_action):
        state = obs['state']
        scan = obs['full_scan']
        n_state = np.concatenate((state, pp_action, s_action, scan))
        self.states.append(n_state)

    def save_states(self, n=0):
        filename = 'SafeData/' + f'run_history_{n}.npy'
        s_data = np.array(self.states)
        np.save(filename, s_data)

        self.states = []


class HistoryManager:
    def __init__(self):
        self.run = None

    def open_history(self, n=0):
        filename = 'SafeData/' + f'run_history_{n}.npy'
        self.run = np.load(filename)

    def step_run(self):
        for set in self.run:
            obs = {}
            obs['full_scan'] = set[-1000::]
            obs['state'] = set[0:5]
            old_action = set[5:7]
            new_action = set[7:9]
            run_safety_check(obs, old_action, 0.4, 3.2, True)
