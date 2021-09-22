import numpy as np
import matplotlib.pyplot as plt

import warnings


def update_state(state, action, dt):
    """
    Updates x, y, th pos accoridng to th_d, v
    """
    L = 0.33
    theta_update = state[2] + ((action[1] / L) * np.tan(action[0]) * dt) 
    dx = np.array([action[1] * np.sin(theta_update),
                action[1]*np.cos(theta_update),
                action[1] / L * np.tan(action[0])])

    return state + dx * dt 


class RotationalObstacle:
    def __init__(self, p1, p2, d_max, n):
        b = 0.05 
        self.op1 = p1 + [-b, -b]
        self.op2 = p2 + [b, -b]
        self.p1 = None
        self.p2 = None
        self.d_max = d_max * 1
        self.obs_n = n
        self.m_pt = np.mean([self.op1, self.op2], axis=0)

        self.ys = []
        self.xs = []
        self.y2s = []
        
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
            return 1, 1

        L = 0.33

        w1 = state_point_x - self.p1[0] 
        w2 = self.p2[0] - state_point_x 

        width_thresh = L / np.tan(self.d_max)

        d1 = np.sqrt(2*L* w1 / np.tan(self.d_max) - w1**2) if w1 < width_thresh else width_thresh
        d2 = np.sqrt(2*L * w2 / np.tan(self.d_max) - w2**2) if w2 < width_thresh else width_thresh

        return d1, d2
  
    def calculate_required_y(self, state):
        d1, d2 = self.find_critical_distances(state[0])

        corrosponding_y = np.interp(state[0], [self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])

        # this is an approximation, but it seems to work. The reason is to make sure that the curvature doesn't breach the line below the end point.
        y1 = np.mean([corrosponding_y, self.p1[1]]) - d1
        y2 = np.mean([corrosponding_y, self.p2[1]]) - d2

        y_safe, d_star = y1, d1
        if y1 < y2:
            y_safe = y2
            d_star = d2

        self.xs.append(state[0])
        self.ys.append(y_safe)
        self.y2s.append(y_safe + d_star)

        return y_safe

    def plot_purity(self):
        pts = np.vstack((self.p1, self.p2))
        plt.plot(pts[:, 0], pts[:, 1], 'x-', markersize=10, color='black')
        pts = np.vstack((self.op1, self.op2))
        plt.plot(pts[:, 0], pts[:, 1], '--', markersize=20, color='black')

    def plot_obstacle(self, state=[0, 0, 0]):
        for i in range(len(self.xs)):
            x = [self.xs[i], self.xs[i]]
            y = [self.ys[i], self.y2s[i]]
            plt.plot(x, y, '-o', color='red', markersize=5)

        self.plot_purity()


def plot_simulated_states(xs, ys, steps=12, theta=0, critical_idx=None):
    if critical_idx is None:
        critical_idx = np.argmin(ys)
    d_max=0.4
    new_states = np.zeros((len(xs), steps, 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        delta = -d_max if i < critical_idx else d_max
        state = [x, y, theta]
        for j in range(steps):
            new_states[i, j] =np.copy(state)
            state = update_state(state, [delta, 1], 0.1)

        plt.plot(new_states[i, :, 0], new_states[i, :, 1], '+-')
    
def slanted_obstacle():
    theta = 0.4
    n_pts = 50

    o = RotationalObstacle(np.array([-0.5,1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.5
    xs = np.linspace(center-width, center+width, n_pts)
    ys = np.zeros((n_pts))
    new_xs = np.zeros(n_pts)

    # skew obstacle
    plt.figure(1)
    plt.title('Straight vehicle, skew obs (transformed state)')
    
    o.transform_obstacle(theta)
    for j, x in enumerate(xs):
        new_xs[j] = x
        ys[j] = o.calculate_required_y([x, 0, theta])

    critical_idx = np.argmin(ys)

    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    plot_simulated_states(xs, ys, theta=0)

    o.plot_obstacle()
    new_pt = o.transform_point([0, 0], theta)
    plt.arrow(new_pt[0], new_pt[1], 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)

    plt.show()


def adaptable_eyes():
    theta = -0.4
    n_pts = 50

    o = RotationalObstacle(np.array([-0.5,1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.5
    xs = np.linspace(center-width, center+width, n_pts)
    ys = np.zeros((n_pts))

    # skew obstacle
    plt.figure(1)
    plt.title('Straight vehicle, skew obs (transformed state)')
    
    o.transform_obstacle(theta)
    for j, x in enumerate(xs):
        ys[j] = o.calculate_required_y([x, 0, theta])

    critical_idx = np.argmin(ys)

    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    plot_simulated_states(xs, ys, theta=0)

    o.plot_obstacle()
    new_pt = o.transform_point([0, 0], theta)
    plt.arrow(new_pt[0], new_pt[1], 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)

    # reverse transform
    plt.figure(2)
    plt.title('Angle vehcile, reverse transformed with orignal obstacle')

    for i in range(n_pts):
        xs[i], ys[i] = o.transform_point([xs[i], ys[i]], -theta)

    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    plot_simulated_states(xs, ys, theta=theta, critical_idx=critical_idx)
    
    o.plot_obstacle()
    plt.arrow(0, 0, 0.2*np.sin(theta), 0.2*np.cos(theta), head_width=0.05)

    plt.show()


def goodbye_singleness():
    theta = -0.4
    n_pts = 30

    o = RotationalObstacle(np.array([-0.5,1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.5
    xs = np.linspace(center-width, center+width, n_pts)
    ys = np.zeros((n_pts))

    o.transform_obstacle(theta)
    o_ys = np.zeros(n_pts)
    for j, x in enumerate(xs):
        o_ys[j] = o.calculate_required_y([x, 0, theta])
        xs[j], ys[j] = o.transform_point([xs[j], o_ys[j]], -theta)

    critical_idx = np.argmin(o_ys)

    # reverse transform
    plt.figure(2)
    plt.title('Angle vehcile, reverse transformed with orignal obstacle')

    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    plot_simulated_states(xs, ys, theta=theta, critical_idx=critical_idx)
    
    o.plot_obstacle()
    plt.arrow(0, 0, 0.2*np.sin(theta), 0.2*np.cos(theta), head_width=0.05)

    plt.show()


if __name__ == '__main__':
    # adaptable_eyes()
    goodbye_singleness()
    # slanted_obstacle()
