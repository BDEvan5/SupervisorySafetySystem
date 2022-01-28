import numpy as np 
import toy_auto_race.Utils.LibFunctions as lib 
import csv 
from matplotlib import pyplot as plt

def save_csv_data(rewards, path):
    data = []
    for i in range(len(rewards)):
        data.append([i, rewards[i]])
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def get_moving_avg(vehicle_name, show=False):
    path = 'Data/PaperVehicles/' + vehicle_name + "/training_data.csv"
    smoothpath = 'Data/PaperVehicles/' + vehicle_name + f"/TrainingData.csv"
    rewards = []
    with open(path, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
        for lines in csvFile:  
            rewards.append(lines)
    rewards = np.array(rewards)[:, 1]

    smooth_rewards = moving_average(rewards, 20)

    # new_rewards = []
    # l = 10
    # N = int(len(smooth_rewards) / l)
    # for i in range(N):
    #     avg = np.mean(smooth_rewards[i*l:(i+1)*l])
    #     new_rewards.append(avg)
    # smooth_rewards = np.array(new_rewards)

    save_csv_data(smooth_rewards, smoothpath)

    if show:
        lib.plot_no_avg(rewards, figure_n=1)
        lib.plot_no_avg(smooth_rewards, figure_n=2)
        plt.show()

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period


get_moving_avg('KernelSSS_1_2', True)