import numpy as np 
from matplotlib import pyplot as plt

def update_state(state, action, dt):
    """
    Updates x, y, th pos accoridng to th_d, v
    """
    L = 0.33
    theta_update = state[2] +  ((action[1] / L) * np.tan(action[0]) * dt)
    dx = np.array([action[1] * np.sin(theta_update),
                action[1]*np.cos(theta_update),
                action[1] / L * np.tan(action[0])])

    return state + dx * dt 


def update_dx(state, action, dt):
    """
    Updates x, y, th pos accoridng to th_d, v
    """
    L = 0.33
    theta_update = state[2] +  ((action[1] / L) * np.tan(action[0]) * dt)
    dx = np.array([action[1] * np.sin(theta_update),
                action[1]*np.cos(theta_update),
                action[1] / L * np.tan(action[0])])

    return dx * dt 


class ObstacleThree:
    L = 0.33

    def __init__(self, p1, p2, d_max=0.4, n=0):
        b = 0.05 
        self.op1 = np.array(p1) + np.array([-b, -b])
        self.op2 = np.array(p2) + np.array([b, -b])
        self.p1 = None
        self.p2 = None
        self.d_max = d_max * 1
        self.obs_n = n
        self.m_pt = np.mean([self.op1, self.op2], axis=0)
        
    def transform_obstacle(self, theta):
        """
        Calculate transformed points based on theta by constructing rotation matrix.
        """
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        self.p1 = rot_m @ (self.op1 - self.m_pt) + self.m_pt
        self.p2 = rot_m @ (self.op2 - self.m_pt) + self.m_pt

    def transform_point(self, pt, theta):
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])

        relative_pt = pt - self.m_pt
        new_pt = rot_m @ relative_pt
        new_pt += self.m_pt
        return new_pt

    def find_critical_distances(self, state_point_x):
        if state_point_x < self.p1[0] or state_point_x > self.p2[0]:
            return 1, 1 #TODO: think about what values to put here

        w1 = state_point_x - self.p1[0] 
        w2 = self.p2[0] - state_point_x 

        width_thresh = self.L / np.tan(self.d_max)

        d1 = np.sqrt(2*self.L* w1 / np.tan(self.d_max) - w1**2) if w1 < width_thresh else width_thresh
        d2 = np.sqrt(2*self.L * w2 / np.tan(self.d_max) - w2**2) if w2 < width_thresh else width_thresh

        return d1, d2
  
    def calculate_required_y(self, x_value):
        d1, d2 = self.find_critical_distances(x_value)
        corrosponding_y = np.interp(x_value, [self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])

        y = min(self.p1[1], self.p2[1])
        y_safe = max(y - d1, y-d2)

        # y1 = corrosponding_y - d1
        # y2 = corrosponding_y - d2
        # y1 = np.mean([corrosponding_y, self.p1[1]]) - d1
        # y2 = np.mean([corrosponding_y, self.p2[1]]) - d2

        # y_safe, d_star = y1, d1
        # if y1 < y2:
        #     y_safe, d_star = y2, d2

        return y_safe

    def plot_obstacle(self):
        scale = 100
        pts = np.vstack((self.p1, self.p2)) * scale
        plt.plot(pts[:, 0], pts[:, 1], 'x-', markersize=10, color='black')
        pts = np.vstack((self.op1, self.op2)) * scale
        plt.plot(pts[:, 0], pts[:, 1], '--', markersize=20, color='black')

    def calculate_safety(self, state=[0, 0, 0]):
        theta = state[2]
        self.transform_obstacle(theta)

        x_search = np.copy(state[0])
        y_safe = self.calculate_required_y(x_search)
        x, y = self.transform_point([x_search, y_safe], -theta)

        print(f"OrigX: {state[0]} -> SearchX: {x_search} -> newX: {x} -> y safe: {y} -> remaining diff: {state[0] - x}")

        while abs(state[0] - x) > 0.01:
            if x_search == self.p1[0] or x_search == self.p2[0]-0.05:
                print(f"Breakin since x_search: {x_search} ->")
                break
            
            x_search = x_search + (state[0]-x)
            x_search = np.clip(x_search, self.p1[0], self.p2[0]-0.05) #TODO check this end condition
            y_safe = self.calculate_required_y(x_search)
            x, y = self.transform_point([x_search, y_safe], -theta)

            print(f"OrigX: {state[0]} -> SearchX: {x_search} -> newX: {x} -> y safe: {y} -> remaining diff: {state[0] - x}")

        return x, y 

    def run_check(self, state):
        x, y = self.calculate_safety(state)

        state_x = state[0]
        if state_x < self.p1[0] or state_x > self.p2[0]:
            return True 
        if state[1] > self.p1[1] and state[1] > self.p2[1]:
            return False

        if y > state[1]:
            return True 
        return False

    def plot_region(self, theta=0):
        # self.transform_obstacle(-0.4)
        # xs = np.linspace(self.p1[0], self.p2[0], 20)
        # ys = np.array([self.calculate_required_y(x) for x in xs])

        # plt.plot(xs, ys, '--', markersize=20, color='black')

        self.transform_obstacle(theta)
        xs = np.linspace(self.p1[0], self.p2[0], 20)
        ys = np.array([self.calculate_required_y(x) for x in xs])

        scale = 100
        plt.plot(xs*scale, ys*scale, '--', markersize=20, color='black')

        # self.transform_obstacle(0.4)
        # xs = np.linspace(self.p1[0], self.p2[0], 20)
        # ys = np.array([self.calculate_required_y(x) for x in xs])

        # plt.plot(xs, ys, '--', markersize=20, color='black')




def run_acting_regions():
    actions = np.ones((5, 2)) * 3
    actions[:, 0] = np.linspace(-0.4, 0.4, 5)
    x_space, y_space = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    obstacle = ObstacleThree([0.2, 1], [0.8, 1])

    # X, Y = np.meshgrid(x_space, y_space)
    # Z = np.zeros((100, 100))

    # for i, x in enumerate(x_space):
    #     for j, y in enumerate(y_space):
    #         state = [x, y, 0]
    #         state = update_state(state, actions[2], 0.1)    
    #         if obstacle.run_check(state):
    #             Z[i, j] = 1


    # fig = plt.figure(1)
    # plt.title(f"Steering: {actions[2]}")
    # plt.imshow(Z.T, origin='lower')
    # obstacle.plot_obstacle()
    # obstacle.plot_region(0)
    # plt.show()

    Z = np.zeros((100, 100))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            state = [x, y, 0]
            state = update_state(state, [0.2, 3], 0.1)    
            # state = update_state(state, actions[0], 0.1)    
            if obstacle.run_check(state):
                Z[i, j] = 1
            if not obstacle.run_check([x, y, 0]):
                Z[i, j] = 2


    fig = plt.figure(2)
    plt.title(f"Steering: {actions[0]}")
    plt.imshow(Z.T, origin='lower')
    obstacle.plot_region(-0.4)
    obstacle.plot_obstacle()
    plt.show()


    # Z = np.zeros((100, 100))
    # for i, x in enumerate(x_space):
    #     for j, y in enumerate(y_space):
    #         state = [x, y, 0]
    #         state = update_state(state, actions[4], 0.1)    
    #         if obstacle.run_check(state):
    #             Z[i, j] = 1


    # fig = plt.figure(3)
    # plt.title(f"Steering: {actions[4]}")
    # plt.imshow(Z.T, origin='lower')
    # obstacle.plot_obstacle()
    # obstacle.plot_region(0.4)
    # plt.show()


if __name__ == "__main__":
    run_acting_regions()

