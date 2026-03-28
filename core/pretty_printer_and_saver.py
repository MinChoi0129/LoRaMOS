import matplotlib.pyplot as plt
import numpy as np
import os


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
    """Prefix rules: GT_/pred_ → class map (0~2), mask_ → heatmap (0~1), feat_ → channel-pooled."""
    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)

    for variable, variable_name in zip(variables, variable_names):
        try:
            single_batch = variable[0].detach().cpu().numpy()
        except:
            single_batch = variable[0].cpu().numpy()

        if variable_name.startswith(("GT_", "pred_")):
            img = single_batch.squeeze()
            plt.imsave(f"{save_dir}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=2)

        elif variable_name.startswith("mask_"):
            img = single_batch.squeeze()
            plt.imsave(f"{save_dir}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=1)

        else:  # feat_ 등
            img = np.mean(single_batch, axis=0) if channel_pool == "mean" else np.max(single_batch, axis=0)
            plt.imsave(f"{save_dir}/{variable_name}.png", img, cmap="viridis")

        print(f"  {variable_name}.png  (min={img.min():.3f}, max={img.max():.3f})")
