import matplotlib.pyplot as plt
import numpy as np
import torch, os


def shprint(*obj):
    n = len(obj)
    print("---------------------------------" * n)
    for o in obj:
        try:
            print(o.shape, end=" | ")
        except:
            print(o, end=" | ")
    print()
    print("---------------------------------" * n)


def save_feature_as_img(variables, variable_names, channel_pool="max"):
    save_dir = f"images"
    os.makedirs(save_dir, exist_ok=True)

    for variable, variable_name in zip(variables, variable_names):
        try:
            single_batch = variable[0].cpu().numpy()
        except:
            single_batch = variable[0].detach().cpu().numpy()

        try:  # feature
            if channel_pool == "mean":
                channel_mean = np.mean(single_batch, axis=0)
                plt.imsave(f"{save_dir}/{variable_name}.png", channel_mean)
            elif channel_pool == "max":
                channel_max = np.max(single_batch, axis=0)
                plt.imsave(f"{save_dir}/{variable_name}.png", channel_max)
            else:
                raise ValueError(f"Invalid channel_pool value: {channel_pool}")
        except:  # label
            plt.imsave(f"{save_dir}/{variable_name}.png", single_batch)
