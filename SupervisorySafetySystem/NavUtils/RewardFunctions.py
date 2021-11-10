import numpy as np

class DistReward:
    def __init__(self):
        self.name = f"Progress"
    # @staticmethod
    def __call__(self, state, s_prime):        
        reward = s_prime['target'][1] - state['target'][1]
        reward += s_prime['reward']

        return reward

class CthReward:
    def __init__(self, b_ct, b_h):
        self.b_ct = b_ct 
        self.b_h = b_h
        self.name = f"Velocity({b_ct})({b_h})"
        self.max_v = 7

    def __call__(self, state, s_prime):        
        # on assumuption of forest with middle @1 and heading =straight 
        pos_x = s_prime['state'][0] 
        reward_ct = abs(1 - pos_x) * self.b_ct 
        scaled_v = s_prime['state'][3] / self.max_v
        reward_h = np.cos(s_prime['state'][2]) * self.b_h * scaled_v

        reward = reward_h - reward_ct
        reward += s_prime['reward']

        return reward


class SteeringReward:
    def __init__(self, b_s):
        self.b_s = b_s
        self.name = f"Steering({b_s})"
        
    def __call__(self, state, s_prime):
        scaled_steering = abs(s_prime['state'][4] / 0.4)
        reward = (0.5 - scaled_steering) * self.b_s
        reward += s_prime['reward']

        return reward


