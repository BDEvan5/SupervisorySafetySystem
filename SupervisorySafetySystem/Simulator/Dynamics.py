from numba import njit
import numpy as np
from matplotlib import pyplot as plt



def update_dynamics(phi, delta, velocity, time_step):
    th_dot = 2 / 0.33 * np.tan(delta)
    new_phi = phi + th_dot * time_step
    dx = np.sin(new_phi) * velocity * time_step
    dy = np.cos(new_phi) * velocity * time_step

    return dx, dy, new_phi


#Dynamics functions
@njit(cache=True)
def update_kinematic_state(x, u, dt, whlb, max_steer, max_v):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    dx = np.array([x[3]*np.sin(x[2]), # x
                x[3]*np.cos(x[2]), # y
                x[3]/whlb * np.tan(x[4]), # theta
                u[1], # velocity
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state


@njit(cache=True)
def control_system(state, action, max_v, max_steer, max_a, max_d_dot):
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
    v_ref = min(action[1], max_v)
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)

    kp_a = 10
    a = (v_ref-state[3])*kp_a
    
    kp_delta = 40
    d_dot = (d_ref-state[4])*kp_delta

    # clip actions
    #TODO: temporary removal of dynamic constraints
    # a = min(a, max_a)
    # a = max(a, -max_a)
    # d_dot = min(d_dot, max_d_dot)
    # d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot, a])

    return u

@njit(cache=True)
def update_complex_state(state, action, dt, plan_steps, whlb, max_steer, max_v):
    for i in range(plan_steps):
        u = control_system(state, action, max_v, max_steer, max_v, max_steer)
        state = update_kinematic_state(state, u, dt, whlb, max_steer, max_v)

    return state


@njit(cache=True)
def update_simple_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    theta_update = x[2] +  ((u[1] / whlb) * np.tan(u[0]) * dt)
    dx = np.array([u[1] * np.sin(theta_update),
                u[1]*np.cos(theta_update),
                u[1] / whlb * np.tan(u[0]),
                u[1],
                u[0]])

    new_state = x + dx * dt 

    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state

@njit(cache=True)
def update_inter_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    theta_update = x[2] +  ((u[1] / whlb) * np.tan(u[0]) * dt)
    dx = np.array([u[1] * np.sin(theta_update),
                u[1]*np.cos(theta_update),
                u[1] / whlb * np.tan(u[0]),
                u[1],
                u[0]])

    new_state = x + dx * dt 

    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state


