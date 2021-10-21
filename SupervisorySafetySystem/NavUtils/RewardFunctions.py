import numpy as np

class DistReward:
    def __init__(self):
        self.name = f"Progress"
    # @staticmethod
    def __call__(self, state, s_prime):        
        reward = s_prime['target'][1] - state['target'][1]
        reward += s_prime['reward']

        return reward

class DistRewardSquare:
    def __init__(self):
        self.name = f"DistSquare"
    # @staticmethod
    def __call__(self, state, s_prime):        
        distance = s_prime['target'][1] - state['target'][1]
        reward = (1+distance)**2 - 1
        reward += s_prime['reward']

        return reward

class DistRewardSqrt:
    def __init__(self):
        self.name = f"DistSqrt"
    # @staticmethod
    def __call__(self, state, s_prime):        
        distance = s_prime['target'][1] - state['target'][1]
        reward = (max(distance, 0))**0.5
        reward += s_prime['reward']

        return reward

class CthReward:
    def __init__(self, b_ct, b_h):
        self.b_ct = b_ct 
        self.b_h = b_h
        self.name = f"Velocity({b_ct})({b_h})"

    def __call__(self, state, s_prime):        
        # on assumuption of forest with middle @1 and heading =straight 
        pos_x = s_prime['state'][0] 
        reward_ct = abs(1 - pos_x) * self.b_ct 
        reward_h = np.cos(s_prime['state'][2]) * self.b_h

        reward = reward_h - reward_ct
        reward += s_prime['reward']

        return reward


class SteeringReward:
    def __init__(self, b_s):
        self.b_s = b_s
        self.name = f"Steering({b_s})"
        
    def __call__(self, state, s_prime):
        reward = - abs(s_prime['state'][4])**0.5 * self.b_s
        reward += s_prime['reward']

        return reward


