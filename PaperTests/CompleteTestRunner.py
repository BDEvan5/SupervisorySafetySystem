from KernelGeneration import * 
from LearningFormulation import *
from BaselineComparision import *
from ClassicalComparision import  *


def main():
    kernel_discretization_ndx()
    kernel_discretization_time()

    generate_standard_kernels()

    rando_pictures()
    rando_results()
    straight_test()


    kernel_reward_tests()
    render_picture()

    eval_continuous(1)
    eval_episodic(1)

    train_baseline_cth(1)
    eval_model_sss(1)

    repeatability_comparision()

    test_FGM()
    test_oracle()


main()
