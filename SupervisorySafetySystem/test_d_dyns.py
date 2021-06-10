
import numpy as np 
from numba import njit
from matplotlib import pyplot as plt


#Dynamics functions
# @njit(cache=True)
def update_kinematic_state(x, u, dt, whlb, max_steer):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    velocity = 3
    dx = np.array([velocity*np.sin(x[2]), # x
                velocity*np.cos(x[2]), # y
                velocity/whlb * np.tan(x[3]), # theta
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[3] = min(new_state[3], max_steer)
    new_state[3] = max(new_state[3], -max_steer)

    return new_state

@njit(cache=True)
def control_system(state, action, max_steer, max_d_dot):
    """
    Generates acceleration and steering velocity commands to follow a reference
    Note: the controller gains are hand tuned in the fcn

    Args:
        v_ref: the reference velocity to be followed
        d_ref: reference steering to be followed

    Returns:
        a: acceleration
        d_dot: the change in delta = steering velocity
    """
    # clip action
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)
    
    kp_delta = 100
    d_dot = (d_ref-state[3])*kp_delta

    d_dot = min(d_dot, max_d_dot)
    d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot])

    return u

# def test():
#     x = np.array([0, 0, 0, 0])
#     a = np.array([0.4, 3])

#     for i in range(10):
#         u = control_system(x, a, 0.4, 3.2)
#         x = update_kinematic_state(x, u, 0.01, 0.33, 0.4)

#         print(f"{i}: State: {x} -> control: {u}")
        
def test():
    d0 = np.linspace(0, 0.4, 100)
    d0 = 0.2
    du = 0.4
    x = np.array([0, 0, 0, d0])
    a = np.array([du, 3])

    for i in range(10):
        u = control_system(x, a, 0.4, 3.2)
        x = update_kinematic_state(x, u, 0.01, 0.33, 0.4)

    # for i in range(1):
    #     u = control_system(x, a, 7, 0.4, 8, 3.2)
    #     x = update_kinematic_state(x, u, 0.1, 0.33, 0.4, 7)

    #     print(f"SingleStep: State: {x} -> control: {u}")

        deg = np.arctan(x[0]/x[1]) * 180 / np.pi 
        print(f"Ref: {a} -> U: {u} --> State: {x} ->  Deg: {deg:.4f} -> th: {x[2]*180/np.pi:.4f}")

    # alpha = np.arcsin(np.tan(d0)*3*0.1/0.66)* 180 / np.pi 
    # beta = np.arcsin(np.tan(du)*3*0.1/0.66)* 180 / np.pi 
    # angle = alpha + beta
    # print(f"Model Angle: {angle} --> alpha: {alpha:.4f} - beta: {beta}")

    # y = 0.33 / np.tan(d0) * np.sin(3 * np.tan(d0)*0.1)
    # x = 0.33 / np.tan(d0) * (1 - np.cos(3 * np.tan(d0)*0.1))

    # th = np.arctan(x / y) * 180 / np.pi 
    # print(f"Model: {th}")

    # x = np.array([0, 0, 0, 3, 0])
    # a = np.array([0.4, 3])

    # for i in range(1):
    #     u = control_system(x, a, 7, 0.4, 8, 3.2)
    #     x = update_kinematic_state(x, u, 0.1, 0.33, 0.4, 7)

    #     print(f"SingleStep: State: {x} -> control: {u}")


def test_d0():
    d0s = np.linspace(0, 0.4, 100)
    degs = np.zeros_like(d0s)
    for j, d0 in enumerate(d0s):
        du = 0.2
        x = np.array([0, 0, 0, 3, d0])
        a = np.array([du, 3])

        for i in range(10):
            u = control_system(x, a, 7, 0.4, 8, 3.2)
            x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

        deg = np.arctan(x[0]/x[1]) * 180 / np.pi 
        print(f"Ref: {a} --> State: {x} ->  Deg: {deg} -> th: {x[2]*180/np.pi}")

        degs[j] = deg

    mod_degs = d0s * 20+ du*6

    plt.figure(1)
    plt.title(f"O steering for input du={du}")
    plt.plot(d0s, degs)
    plt.plot(d0s, mod_degs)
    plt.xlabel('O steering')
    plt.show()
    plt.pause(0.0001)


def test_du():
    dus = np.linspace(0, 0.4, 100)
    degs = np.zeros_like(dus)
    for j, du in enumerate(dus):
        d0 = 0
        x = np.array([0, 0, 0, 3, d0])
        a = np.array([du, 3])

        for i in range(10):
            u = control_system(x, a, 7, 0.4, 8, 3.2)
            x = update_kinematic_state(x, u, 0.01, 0.33, 0.4, 7)

        deg = np.arctan(x[0]/x[1]) * 180 / np.pi 
        print(f"Ref: {a} --> State: {x} ->  Deg: {deg} -> th: {x[2]*180/np.pi}")

        degs[j] = deg

    # mod_degs = dus * 5 + d0 * 20
    mod_degs = 0.02* 150000**dus + d0 * 19

    plt.figure(2)
    plt.title(f'Du: input steering for: d0={d0}')
    plt.plot(dus, degs)
    plt.plot(dus, mod_degs)
    plt.xlabel('Input steering')
    plt.show()


if __name__ == "__main__":
    test()
    # test_d0()
    # test_du()
