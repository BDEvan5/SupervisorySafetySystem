from SupervisorySafetySystem.KenelTests.ConstructingKernel import DiscriminatingImgKernel, Kernel, SafetyPlannerPP, KernelSim

from matplotlib import pyplot as plt
import numpy as np

def run_test_loop(env, planner, show=False, n_tests=100):
    success = 0

    for i in range(n_tests):
        done = False
        state = env.reset()
        planner.kernel.construct_kernel(env.env_map.map_img.shape, env.env_map.obs_pts)
        while not done:
            a = planner.plan(state)
            s_p, r, done, _ = env.step(a)
            state = s_p

        if r == -1:
            print(f"{i}: Crashed -> {s_p['state']}")
        elif r == 1:
            print(f"{i}: Success")
            success += 1 

        if show:
            env.render_ep()
            plt.pause(0.5)

            if r == -1:
                plt.show()

    print("Success rate: {}".format(success/100))


def construct_obs_kernel(obs_kernel_name, resolution):
    img_size = 1.3 # size in meters 
    obs_size = 0.5 # size in meters
    img_size = int(img_size * resolution)
    obs_size = int(obs_size * resolution)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = DiscriminatingImgKernel(img)
    kernel.calculate_kernel()
    kernel.save_kernel(obs_kernel_name)

def constructy_kernel_sides(side_kernel_name, resolution):
    img_size = np.array([2, 1] , dtype=int) * resolution # size in meters
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = DiscriminatingImgKernel(img)
    kernel.calculate_kernel()
    kernel.save_kernel(side_kernel_name)


def std_test():
    side_name = "std_side_kernel"
    obs_name = "std_obs_kernel"
    resolution = 100 
    constructy_kernel_sides(side_name, resolution)
    construct_obs_kernel(obs_name, resolution)

    env = KernelSim()  
    planner = SafetyPlannerPP()
    planner.kernel = Kernel(side_name, obs_name, resolution)

    run_test_loop(env, planner, False, 10)






if __name__ == "__main__":
    std_test()


