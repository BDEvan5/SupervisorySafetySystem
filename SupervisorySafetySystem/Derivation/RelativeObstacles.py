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
        print(self.m_pt)

        self.ys = []
        self.xs = []
        self.y2s = []
        
    def transform_obstacle_line(self, theta):
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

        # make sure that you are rotating around the correct point
        relative_pt = pt - self.m_pt
        new_pt = rot_m @ relative_pt
        new_pt += self.m_pt
        return new_pt

    def find_critical_distances(self, state_point_x):
        """
        this function takes a point that has been transformed to have theta = 0. i.e. the point is facing straight up and the obstacle has been adjusted. 
        """
        if state_point_x < self.p1[0] or state_point_x > self.p2[0]:
            return 1, 1

        L = 0.33

        w1 = state_point_x - self.p1[0] 
        w2 = self.p2[0] - state_point_x 

        width_thresh = L / np.tan(self.d_max)

        d1 = np.sqrt(2*L* w1 / np.tan(self.d_max) - w1**2) if w1 < width_thresh else width_thresh
        d2 = np.sqrt(2*L * w2 / np.tan(self.d_max) - w2**2) if w2 < width_thresh else width_thresh

        return d1, d2
  
    # def transform_distances(self, d1, d2, theta):
    #     vector = np.array([])
        # plt.plot(pts[:, 0], pts[:, 1], '--', markersize=20, color='black')

    def calculate_required_y(self, state):
        # run transform first
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

        # pt = np.array([state[0], y_safe])
        # new_pt = self.transform_point(pt, -state[2])

        # return new_pt

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

class ObstacleThree:
    def __init__(self, p1, p2, d_max, n):
        b = 0.05 
        self.op1 = p1 + [-b, -b]
        self.op2 = p2 + [b, -b]
        self.p1 = None
        self.p2 = None
        self.d_max = d_max * 1
        self.obs_n = n

    def run_check(self, state):
        pt = self.calculate_transforms(state) # calculates transformed pts
        
        # check if the obs is in front of pt. #TODO: check all these if statements. It isn't picking up is the point is in front or not. 
        if pt[0] < self.p1[0] or pt[0] > self.p2[0]:
            if self.p1[0] < 0 and self.p2[0] > 0:
                if pt[0] < self.p1[0] and self.p1[0] < 0:
                    if pt[1] / pt[0] < self.p1[1] / self.p1[0] :
                        return False
                if pt[0] < self.p2[0] and self.p2[0] > 0:
                    if pt[1] / pt[0] > self.p2[1] / self.p2[0]:
                        return False     

            return True 
        if pt[1] > self.p1[1] and pt[1] > self.p2[1]:
            return False

        y_required = self.find_critical_point(pt)


        if y_required > pt[1]:
            safe_value = True 
        else:
            safe_value = False

        # print(f"{safe_value}: Obs{self.obs_n} -> y_req:{y_required:.4f}, NewPt: {pt} ->start:{self.p1}, end: {self.p2}")

        return safe_value

    def calculate_transforms(self, theta):
        """
        Calculate transformed points based on theta by constructing rotation matrix.
        """
        # theta = state[2]
        m_pt = [0, 1]
        rot_m = np.array([[np.cos(theta), -np.sin(theta)], 
                [np.sin(theta), np.cos(theta)]])
        self.p1 = rot_m @ (self.op1 - m_pt) + m_pt
        self.p2 = rot_m @ (self.op2 - m_pt) + m_pt

        # new_pt = rot_m @ state[0:2]


        # return new_pt

    def reverse_rotate(self, pt, theta):
        rot_m = np.array([[np.cos(-theta), -np.sin(-theta)], 
                [np.sin(-theta), np.cos(-theta)]])

        new_pt = rot_m @ pt
        return new_pt

    def plot_obstacle(self, state=[0, 0, 0]):
        pts = np.vstack((self.p1, self.p2))
        plt.plot(pts[:, 0], pts[:, 1], 'x-', markersize=10, color='black')
        pts = np.vstack((self.op1, self.op2))
        plt.plot(pts[:, 0], pts[:, 1], '--', markersize=20, color='black')

    def find_critical_point(self, state_point_x):
        """
        this function takes a point that has been transformed to have theta =0. i.e. the point is facing straight up and the obstacle has been adjusted. 
        """
        if state_point_x < self.p1[0] or state_point_x > self.p2[0]:
            return 1

        L = 0.33

        w1 = state_point_x - self.p1[0] 
        w2 = self.p2[0] - state_point_x 

        width_thresh = L / np.tan(self.d_max)

        # if w1 < width_thresh and w2 < width_thresh:
        #     return 1
            
        # if w1 > width_thresh or w2 > width_thresh: #TODO: the or isn't working. If one doesn't work, use the other one. 
        #     return 1


        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                d1 = np.sqrt(2*L* w1 / np.tan(self.d_max) - w1**2)
                d2 = np.sqrt(2*L * w2 / np.tan(self.d_max) - w2**2)
            except RuntimeWarning as e:
                print(f"Warning caught: p1: {self.p1} -> p2: {self.p2} -> w1,2: {w1},{w2}, state: {state_point_x}")
                print(e)
                raise

        y1 = self.p1[1] - d1
        y2 = self.p2[1] - d2

        y_safe = max(y1, y2)
        return y_safe 
  

def run_relative_relationship():
    o = ObstacleThree(np.array([-0.5, 1]), np.array([0.5, 1]), 0.4, 1)
    n_angles = 20 
    n_xs = 20
    angles = np.linspace(-np.pi/4, np.pi/4, n_angles)
    xs = np.linspace(-0.7, 0.7, 20)

    ys = np.zeros((n_angles, n_xs))
    
    for i, theta in enumerate(angles):
        o.calculate_transforms(theta)
        for j, x in enumerate(xs):
            ys[i, j] = o.find_critical_point(x)

    fig = plt.figure()
 
    ax = plt.axes(projection ='3d')
    
    X, Z = np.meshgrid(xs, angles)
    ax.plot_surface(X, ys, Z, cmap='viridis', rstride=1, cstride=1)
    ax.set_title('3D line plot geeks for geeks')
    ax.set_xlabel('X axis: x value input')
    ax.set_ylabel('Y axis: v value returned')
    ax.set_zlabel('Z axis: angle')
    plt.show()

def finding_jellyfish_points(theta, n_xs=20):

    o = ObstacleThree(np.array([-0.5, 1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.8
    xs = np.linspace(center-width, center+width, n_xs)
    ys = np.zeros((n_xs))
    
    o.calculate_transforms(theta)
    for j, x in enumerate(xs):
        ys[j] = o.find_critical_point(x)

    return xs, ys, o


def making_magdaleen():
    xs, ys = finding_jellyfish_points()

    plt.figure()
    plt.plot(xs, ys)
    plt.ylim([-0.2, 1.2])
    plt.show()

def cutting_cucumbers():
    n_angles = 10
    n_pts = 20
    th_lim = 0.4
    angles = np.linspace(-th_lim, th_lim, n_angles)
    zss = np.zeros((n_angles, 20))
    xss = np.zeros((n_angles, 20))
    yss = np.zeros((n_angles, 20))
    for i in range(n_angles):
        xss[i], yss[i] = finding_jellyfish_points(angles[i])
    # zss = angles[i] * np.ones(n_pts)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    X, Y = np.meshgrid(xss[0], angles)
    ax.plot_surface(X, Y, yss, cmap='viridis', rstride=1, cstride=1)
    ax.set_xlabel('X axis: z input')
    ax.set_ylabel('Y axis: angle')
    ax.set_zlabel('Z axis: y output')
    plt.show()

    plt.figure()
    plt.plot(xss[0], yss[0])

    plt.figure()
    plt.plot(xss[1], yss[1])

    plt.figure()
    plt.plot(xss[2], yss[2])

    plt.show()
    
def adaptable_eyes():
    theta = 0.4
    n_pts = 50
    d_max = 0.4
    steps = 12 

    # skew obstacle
    plt.figure(1)
    plt.title('Straight vehicle, skew obs (transformed state)')
    o = RotationalObstacle(np.array([-0.5,1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.5
    xs = np.linspace(center-width, center+width, n_pts)
    ys = np.zeros((n_pts))
    new_xs = np.zeros(n_pts)
    
    o.transform_obstacle_line(theta)
    for j, x in enumerate(xs):
        new_xs[j] = x
        ys[j] = o.calculate_required_y([x, 0, theta])

    critical_idx = np.argmin(ys)
    # critical_x = xs[np.argmin(ys)]

    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    new_states = np.zeros((n_pts, steps, 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        delta = -d_max if i < critical_idx else d_max
        # delta = d_max
        state = [x, y, 0]
        for j in range(steps):
            new_states[i, j] =np.copy(state)
            state = update_state(state, [delta, 1], 0.1)

    
    for i in range(n_pts):
        plt.plot(new_states[i, :, 0], new_states[i, :, 1], '+-')

    o.plot_obstacle()

    # reverse transform
    plt.figure(2)
    plt.title('Angle vehcile, reverse transformed with orignal obstacle')

    
    for i in range(n_pts):
        xs[i], ys[i] = o.transform_point([xs[i], ys[i]], -theta)

    # critical_idx = np.argmin(ys)
    
    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    new_states = np.zeros((n_pts, steps, 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        delta = -d_max if i < critical_idx else d_max
        # delta = d_max
        state = [x, y, theta]
        for j in range(steps):
            new_states[i, j] =np.copy(state)
            state = update_state(state, [delta, 1], 0.1)

    for i in range(n_pts):
        plt.plot(new_states[i, :, 0], new_states[i, :, 1], '+-')

    new_pt = o.transform_point([0, 0], theta)

    o.plot_obstacle()
    plt.arrow(0, 0, 0.2*np.sin(theta), 0.2*np.cos(theta), head_width=0.05)
    plt.arrow(new_pt[0], new_pt[1], 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)

    plt.show()

    
def walking_to_love():
    theta = 0.4
    n_pts = 35
    d_max = 0.4
    steps = 12 
    # xs, ys, o = finding_jellyfish_points(theta, n_pts)
    xs, ys, o = slanted_squid(theta, 20)
    critical_x = xs[np.argmin(ys)]

    new_states = np.zeros((n_pts, steps, 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        delta = -d_max if x < critical_x else d_max
        # delta = d_max
        state = [x, y, theta]
        for j in range(steps):
            new_states[i, j] =np.copy(state)
            state = update_state(state, [delta, 1], 0.1)

    plt.figure()
    plt.plot(xs, ys, linewidth=2)
    plt.ylim([-0.2, 1.2])

    for i in range(n_pts):
        plt.plot(new_states[i, :, 0], new_states[i, :, 1], '+-')

    new_pt = o.transform_point([0, 0], theta)

    o.plot_obstacle()
    plt.arrow(0, 0, 0.2*np.sin(theta), 0.2*np.cos(theta), head_width=0.05)
    plt.arrow(new_pt[0], new_pt[1], 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)

    plt.show()

def a_heart_of_buter():
    theta = 0.4
    n_pts = 10
    d_max = 0.4
    steps = 12

    xs = np.zeros(n_pts)
    ys = np.linspace(0.1, 0.6, n_pts)
    o = ObstacleThree(np.array([-0.5, 1]), np.array([0.5, 1]), 0.4, 1)
    o.calculate_transforms(theta)
    Y = o.find_critical_point(0)

    # pt = o.reverse_rotate([0, Y], theta)

    new_states = np.zeros((n_pts, steps, 3))
    for i, (x, y) in enumerate(zip(xs, ys)):
        # delta = -d_max if x < critical_x else d_max
        state = [x, y, theta]
        for j in range(steps):
            new_states[i, j] =np.copy(state)
            state = update_state(state, [d_max, 1], 0.1)

    plt.figure()
    plt.plot(0, Y, 'x', markersize=20)
    # plt.plot(pt[0], pt[1], 'x', markersize=20)
    plt.ylim([-0.2, 1.2])

    for i in range(n_pts):
        plt.plot(new_states[i, :, 0], new_states[i, :, 1], '+-')

    o.plot_obstacle()

    plt.show()

def slanted_squid(theta, n_xs=20):
    # o = RotationalObstacle(np.array([-0.5,0.8]), np.array([0.5, 1.2]), 0.4, 1)
    o = RotationalObstacle(np.array([-0.5,1]), np.array([0.5, 1]), 0.4, 1)
    center = -0
    width = 0.5
    xs = np.linspace(center-width, center+width, n_xs)
    ys = np.zeros((n_xs))
    new_xs = np.zeros(n_xs)
    
    o.transform_obstacle_line(theta)
    for j, x in enumerate(xs):
        new_xs[j] = x
        ys[j] = o.calculate_required_y([x, 0, theta])


    # plt.figure()
    # plt.plot(xs, ys)
    # o.plot_obstacle()
    # plt.ylim([-0.2, 1.2])
    # plt.show()

    return xs, ys, o

def talking_in_circles():
    theta = np.pi/8

    o = RotationalObstacle(np.array([-0.5, 1]), np.array([0.5, 1]), 0.4, 1)
    o.transform_obstacle_line(theta)

    l = 0.2
    new_pt = o.transform_point([0, 0], theta)

    plt.figure()
    o.plot_purity()
    plt.arrow(0, 0, 0.2*np.sin(theta), 0.2*np.cos(theta), head_width=0.05)
    # plt.arrow(0, 0, 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)
    plt.arrow(new_pt[0], new_pt[1], 0.2*np.sin(0), 0.2*np.cos(0), head_width=0.05)
    plt.show()


if __name__ == '__main__':
    # run_relative_relationship()
    # making_magdaleen()
    # cutting_cucumbers()
    # walking_to_love()
    # a_heart_of_buter()
    # slanted_squid()

    # talking_in_circles()

    adaptable_eyes()

