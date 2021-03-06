import numpy as np
from SandboxSafety.Simulator.Dynamics import update_complex_state


class Modes:
    def __init__(self, sim_conf):
        self.time_step = sim_conf.time_step
        self.nq_steer = sim_conf.nq_steer
        self.nq_velocity = sim_conf.nq_velocity
        self.max_steer = sim_conf.max_steer
        self.max_velocity = sim_conf.max_v
        self.min_velocity = sim_conf.min_v

        self.vs = np.linspace(self.min_velocity, self.max_velocity, self.nq_velocity)
        self.ds = np.linspace(-self.max_steer, self.max_steer, self.nq_steer)

        self.qs = None
        self.n_modes = None
        self.nv_modes = None
        self.v_mode_list = None
        self.nv_level_modes = None
        self.actions = None
        self.v_res = (self.max_velocity - self.min_velocity) / (self.nq_velocity - 1)

        self.init_modes()
        # self.built_transition_actions()

    def init_modes(self):
        b = 0.523
        g = 9.81
        l_d = 0.329

        mode_list = []
        v_mode_list = []
        nv_modes = [0]
        for i, v in enumerate(self.vs):
            v_mode_list.append([])
            for s in self.ds:
                if abs(s) < 0.06:
                    mode_list.append([s, v])
                    v_mode_list[i].append(s)
                    continue

                friction_v = np.sqrt(b*g*l_d/np.tan(abs(s))) *1.1 # nice for the maths, but a bit wrong for actual friction
                if friction_v > v:
                    mode_list.append([s, v])
                    v_mode_list[i].append(s)

            nv_modes.append(len(v_mode_list[i])+nv_modes[-1])

        self.qs = np.array(mode_list) # modes1
        self.n_modes = len(mode_list) # n modes
        self.nv_modes = np.array(nv_modes) # number of v modes in each level
        self.nv_level_modes = np.diff(self.nv_modes) # number of v modes in each level
        self.v_mode_list = v_mode_list # list of steering angles sorted by velocity
        for i in range(len(self.v_mode_list)):
            self.v_mode_list[i] = np.array(self.v_mode_list[i])

    def get_mode_id(self, v, d):
        # assume that a valid input is given that is within the range.
        v_ind = np.argmin(np.abs(self.vs - v))
        d_ind = np.argmin(np.abs(self.v_mode_list[v_ind] - d))
        
        return_mode = self.nv_modes[v_ind] + d_ind
        
        return int(return_mode)

    def action2mode(self, action):
        id = self.get_mode_id(action[1], action[0])
        return self.qs[id]

    def check_state_modes(self, v, d):
        b = 0.523
        g = 9.81
        l_d = 0.329
        if abs(d) < 0.06:
            return True # safe because steering is small
        friction_v = np.sqrt(b*g*l_d/np.tan(abs(d))) *1.1 # nice for the maths, but a bit wrong for actual friction
        if friction_v > v:
            return True # this is allowed mode
        return False # this is not allowed mode: the friction is too high

    def get_safe_mode_id(self, v, d):
        if not self.check_state_modes(v, d):
            return None

        # a valid input is now guaranteed
        v_ind = np.argmin(np.abs(self.vs - v))
        d_ind = np.argmin(np.abs(self.v_mode_list[v_ind] - d))
        
        return_mode = self.nv_modes[v_ind] + d_ind
        
        return return_mode

    def __len__(self): return self.n_modes

    def built_transition_actions(self): 
        actions = []
        for s, state_mode in enumerate(self.qs):
            qas = []
            state = np.array([0, 0, 0, state_mode[1], state_mode[0]])
            for i, qact in enumerate(self.qs):
                new_state = update_complex_state(state, qact, self.time_step)
                dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
                new_q = self.get_safe_mode_id(vel, steer)

                print(f"State: {s},{state_mode} + Action: {i},{qact} --> new_q: {new_q}, [{steer:.3f}  {vel:.3f}]")
                if new_q is not None:
                    qas.append(i)
            actions.append(qas)
        self.actions = actions
        print("Transition actions built")
        print(f"Actions: {actions}")

    def get_allowed_actions(self, state_mode):
        """
        This method is to see what action modes are allowed for a certain state. It monitors the dynamic updates which are position and orienttaion independant becuase the velocity and steering are the only thing of concern.

        For a given state mode, what actions will actually have an effect? 
        """
        return self.actions[state_mode]


from argparse import Namespace
import yaml
def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


def test_modes():
    conf = load_conf("PaperKernelGen")
    # conf = load_conf("std_test_kernel")
    m = Modes(conf)
    m.init_modes()
    
    l = m.qs.copy()
    for q in l:
        print(f"{q} -> {m.get_mode_id(q[1], q[0])}")


if __name__ == "__main__":
    # test_q_fcns()
#     conf = load_conf("track_kernel")
#     build_track_kernel(conf)

    test_modes()
    
