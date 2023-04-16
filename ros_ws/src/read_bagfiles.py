from bagpy import bagreader
import bagpy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


b = bagreader('test_before_easter.bag')


# get the list of topics
print(b.topic_table)

csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

# csvfiles is now a list of csvfiles, one for each topic
# you can now use pandas to read the csv files and plot them


# 6-dimensional. Each term represents the angle of a thruster
df_alpha = pd.read_csv(csvfiles[0])
# 15-dimensional. Represents observer states [eta, nu, bias, xi]
df_obs = pd.read_csv(csvfiles[1])
# 6-dimensional
df_ref = pd.read_csv(csvfiles[2])
# 6-
df_tau_cmd = pd.read_csv(csvfiles[3])
#
df_u = pd.read_csv(csvfiles[4])
#

print(df_tau_cmd)


def plot_alpha():
    plt.figure()
    for i in range(6):
        plt.plot(df_alpha["data_"+str(i)], label="data_"+str(i))
    plt.legend()
    plt.show()


def plot_obs():
    eta = np.array([df_obs["data_0"], df_obs["data_1"], df_obs["data_2"]])
    nu = np.array([df_obs["data_3"], df_obs["data_4"], df_obs["data_5"]])
    bias = np.array([df_obs["data_6"], df_obs["data_7"], df_obs["data_8"]])
    xi = np.array([df_obs["data_9"], df_obs["data_10"], df_obs["data_11"],
                  df_obs["data_12"], df_obs["data_13"], df_obs["data_14"]])
    dof = ['surge', 'sway', 'yaw']

    fig, axs = plt.subplots(3, 1)
    plt.suptitle('Observer position', fontsize=24)
    for i in range(3):
        axs[i].plot(eta[i])
        axs[i].set_ylabel(r"$\eta_{"+dof[i]+"}$"+" [m]")
        axs[i].set_title(dof[i])
    axs[2].set_xlabel("Time")
    fig.set_tight_layout(True)

    fig, axs = plt.subplots(3, 1)
    plt.suptitle('Observer velocity', fontsize=24)
    for i in range(3):
        axs[i].plot(nu[i])
        axs[i].set_ylabel(r"$\nu_{"+dof[i]+"}$"+" [m/s]")
        axs[i].set_title(dof[i])
    axs[2].set_xlabel("Time")
    fig.set_tight_layout(True)

    fig, axs = plt.subplots(3, 1)
    plt.suptitle('Observer bias', fontsize=24)
    for i in range(3):
        axs[i].plot(bias[i])
        axs[i].set_ylabel(r"$b{"+dof[i]+"}$"+" [N]")
        axs[i].set_title(dof[i])
    axs[2].set_xlabel("Time")
    fig.set_tight_layout(True)

    fig, axs = plt.subplots(2, 1)
    plt.suptitle('Observer wave', fontsize=24)
    for i in range(2):
        for j in range(3):
            axs[i].plot(xi[i*3+j])
            axs[i].set_ylabel(r"$\xi_{"+dof[j]+"}$")
            axs[i].set_title(dof[j])
            axs[i].legend()
    axs[1].set_xlabel("Time")
    fig.set_tight_layout(True)

    plt.figure(figsize=(4, 4))
    plt.title("Observer trajectory")
    plt.plot(eta[1], eta[0], label='Observer trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.show()
