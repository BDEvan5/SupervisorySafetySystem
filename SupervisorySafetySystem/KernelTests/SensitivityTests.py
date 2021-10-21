from SupervisorySafetySystem.KernelTests.ConstructingKernel import DiscriminatingImgKernel, Kernel, SafetyPlannerPP, KernelSim

from matplotlib import pyplot as plt
import numpy as np
from argparse import Namespace
import yaml

def load_conf(fname):
    full_path =  "config/" + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


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

    print("Success rate: {}".format(success/n_tests))

    return success/n_tests


def construct_obs_kernel(conf):
    img_size = int(conf.obs_img_size / conf.resolution)
    obs_size = int(conf.obs_size / conf.resolution)
    obs_offset = int((img_size - obs_size) / 2)
    img = np.zeros((img_size, img_size))
    img[obs_offset:obs_size+obs_offset, -obs_size:-1] = 1 
    kernel = DiscriminatingImgKernel(img)
    kernel.calculate_kernel()
    kernel.save_kernel(f"ObsKernel_{conf.kernel_name}")

def constructy_kernel_sides(conf): #TODO: combine to single fcn?
    img_size = np.array(np.array(conf.side_img_size) / conf.resolution , dtype=int) 
    img = np.zeros(img_size) # use res arg and set length
    img[0, :] = 1
    img[-1, :] = 1
    kernel = DiscriminatingImgKernel(img)
    kernel.calculate_kernel()
    kernel.save_kernel(f"SideKernel_{conf.kernel_name}")



def std_test():
    conf = load_conf("kernel_config")

    constructy_kernel_sides(conf)
    construct_obs_kernel(conf)

    env = KernelSim(conf)  
    planner = SafetyPlannerPP()
    planner.kernel = Kernel(conf)

    run_test_loop(env, planner, False, 10)

def disretization_test():
    resolutions = [50, 80, 100, 120]
    results = np.zeros_like(resolutions)
    for i, resolution in enumerate(resolutions):
        print(f"Running discretisation test: {resolution}")
        side_name = f"discret_side_kernel_{resolution}"
        obs_name = f"discret_obs_kernel_{resolution}"
        constructy_kernel_sides(side_name, resolution)
        construct_obs_kernel(obs_name, resolution)

        env = KernelSim()  
        planner = SafetyPlannerPP()
        planner.kernel = Kernel(side_name, obs_name, resolution)

        results[i] = run_test_loop(env, planner, False, 10)

    print(resolutions)
    print(results)  

if __name__ == "__main__":
    std_test()
    # disretization_test()


