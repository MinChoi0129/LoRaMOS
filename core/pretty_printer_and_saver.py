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
        # 3D 포인트클라우드: (xyz, logit, valid_mask) tuple → [N_valid, 4] npy 저장
        if variable_name.startswith("pcd_"):
            xyz_t0, logit_3d, valid_t0 = variable
            xyz = xyz_t0[0].detach().cpu().numpy()  # [3, N]
            max_logit = logit_3d[0].squeeze(-1).detach().cpu()
            max_logit = max_logit.max(dim=0)[0].numpy()  # [N]
            mask = valid_t0[0].detach().cpu().numpy() > 0.5  # [N]
            data = np.column_stack([xyz[:, mask].T, max_logit[mask]]).astype(np.float32)
            print(f"Saving :  {variable_name} as pointcloud npy ({data.shape[0]} points, {data.shape[1]} cols)")
            np.save(f"{save_dir}/{variable_name}.npy", data)
            continue

        try:
            single_batch = variable[0].cpu().numpy()
        except:
            single_batch = variable[0].detach().cpu().numpy()

        if variable_name.startswith("GT_") or variable_name.startswith("pred_"):
            img = single_batch.squeeze()
            print("Saving : ", variable_name, "as class map")
            plt.imsave(f"{save_dir}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=2)
        elif variable_name.startswith("heatmap_"):
            img = single_batch.squeeze()
            print("Saving : ", variable_name, "as heatmap")
            plt.imsave(f"{save_dir}/{variable_name}.png", img, cmap="viridis", vmin=0, vmax=1)
        else:
            if channel_pool == "mean":
                pooled = np.mean(single_batch, axis=0)
            else:
                pooled = np.max(single_batch, axis=0)
            print("Saving : ", variable_name, "as feature")
            plt.imsave(f"{save_dir}/{variable_name}.png", pooled, cmap="viridis")
