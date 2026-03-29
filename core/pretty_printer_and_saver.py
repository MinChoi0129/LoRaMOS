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


def save_feature_as_img(items, names=None, channel_pool="max"):
    """Prefix rules: GT_/pred_ → class map (0~2), mask_ → heatmap (0~1), feat_ → channel-pooled.
    Items can be:
      - 2-tuple (tensor, name): 2D image → save as .png
      - 3-tuple (xyz, value, name): 3D point cloud → save as .npy (N, 4)
    If names is provided (legacy), items should be a list of tensors.
    """
    dir_2d = "visualization/2d"
    dir_3d = "visualization/3d"
    os.makedirs(dir_2d, exist_ok=True)
    os.makedirs(dir_3d, exist_ok=True)

    # Support legacy call: save_feature_as_img([tensors], [names], pool)
    if names is not None:
        items = list(zip(items, names))

    for item in items:
        if len(item) == 3 and isinstance(item[2], str):
            # 3D point cloud: (xyz [B, 3, N], value [B, C, N], name)
            xyz, value, name = item
            xyz_np = xyz[0].detach().cpu().numpy()        # [3, N]
            val_np = value[0].detach().cpu().numpy()       # [C, N]

            if name.startswith("feat_"):
                scalar = np.max(val_np, axis=0)            # channel-max → [N]
            elif name.startswith("pred_"):
                scalar = val_np.squeeze(0)                 # [N]
            else:  # logit_
                scalar = np.max(val_np, axis=0)            # max logit → [N]

            pcd = np.stack([xyz_np[0], xyz_np[1], xyz_np[2], scalar], axis=1)  # [N, 4]
            np.save(f"{dir_3d}/{name}.npy", pcd)
            print(f"  {name}.npy  shape={pcd.shape} (val min={scalar.min():.3f}, max={scalar.max():.3f})")

        else:
            # 2D image: (tensor, name)
            variable, variable_name = item
            try:
                single_batch = variable[0].detach().cpu().numpy()
            except:
                single_batch = variable[0].cpu().numpy()

            if variable_name.startswith(("GT_", "pred_")):
                img = single_batch.squeeze()
                plt.imsave(f"{dir_2d}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=2)

            elif variable_name.startswith("mask_"):
                img = single_batch.squeeze()
                plt.imsave(f"{dir_2d}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=1)

            else:  # feat_ 등
                img = np.mean(single_batch, axis=0) if channel_pool == "mean" else np.max(single_batch, axis=0)
                plt.imsave(f"{dir_2d}/{variable_name}.png", img, cmap="viridis")

            print(f"  {variable_name}.png  (min={img.min():.3f}, max={img.max():.3f})")
