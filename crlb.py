import os
import torch
import numpy as np
import time

def get_r_matrix(fov_arg):
    R = torch.zeros((fov_arg.pixel_num, fov_arg.pixel_num), dtype=torch.float32)

    for idx_x in range(fov_arg.pixel_num_x):
        for idx_y in range(fov_arg.pixel_num_y):
            for idx_z in range(fov_arg.pixel_num_z):
                idx_k = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + idx_x
                # self
                R[idx_k, idx_k] = 4

                # neighbour of x
                if idx_x > 0 and idx_x + 1 < fov_arg.pixel_num_x:
                    idx_l_1 = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + (idx_x + 1)
                    idx_l_2 = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + (idx_x - 1)
                    R[idx_k, idx_l_1] = -1
                    R[idx_k, idx_l_2] = -1

                # neighbour of y
                if idx_y > 0 and idx_y + 1 < fov_arg.pixel_num_y:
                    idx_l_1 = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + (idx_y + 1) * fov_arg.pixel_num_x + idx_x
                    idx_l_2 = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + (idx_y - 1) * fov_arg.pixel_num_x + idx_x
                    R[idx_k, idx_l_1] = -1
                    R[idx_k, idx_l_2] = -1

                if idx_z > 0 and idx_z + 1 < fov_arg.pixel_num_z:
                    idx_l_1 = (idx_z + 1) * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + idx_x
                    idx_l_2 = (idx_z - 1) * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + idx_x
                    R[idx_k, idx_l_1] = -1
                    R[idx_k, idx_l_2] = -1

    return R

def get_crc_var(R, img, eval_arg, fov_arg, save_path, name_sys, name_val, device):
    print("----", name_val, "starts----")
    sysmat_file_path = "./SysMat/" + name_sys + "/" + name_val
    sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32),[fov_arg.pixel_num, -1])).transpose(0, 1)
    sysmat = sysmat.to(device, non_blocking=True)

    # Calculate average sensitivity（uniform phantom over whole fov）
    avg_sens = torch.sum(torch.matmul(sysmat, img)).item() / fov_arg.pixel_num
    print(f"Average Sensitivity = {avg_sens:.6f}")

    # Calculate FIM
    proj = torch.matmul(sysmat, img)
    proj_inv = proj.reciprocal()
    tmp_sysmat = sysmat * proj_inv
    FIM = torch.matmul(sysmat.t(), tmp_sysmat)

    # results container
    num_levels = len(eval_arg.eval_beta)
    CRC_points = torch.zeros((num_levels, len(eval_arg.eval_point_idx)), dtype=torch.float32)
    Var_points = torch.zeros_like(CRC_points)

    CRC_mean = torch.zeros(num_levels, dtype=torch.float32)
    CRC_std = torch.zeros_like(CRC_mean)
    Var_mean = torch.zeros_like(CRC_mean)
    Var_std = torch.zeros_like(CRC_mean)

    for i, beta in enumerate(eval_arg.eval_beta):
        # calculate LIR COV
        A = FIM + beta * R
        FIM_inv = torch.inverse(A)
        LIR = FIM_inv.matmul(FIM)
        Cov = LIR.matmul(FIM_inv)

        # ROI 内部像素统计
        CRC_list = []
        Var_list = []

        for idx_x in range(fov_arg.pixel_num_x):
            for idx_y in range(fov_arg.pixel_num_y):
                for idx_z in range(fov_arg.pixel_num_z):
                    pos_x = (idx_x - fov_arg.pixel_num_x / 2 + 0.5) * fov_arg.pixel_l_x
                    pos_y = (idx_y - fov_arg.pixel_num_y / 2 + 0.5) * fov_arg.pixel_l_y
                    pos_z = (idx_z - fov_arg.pixel_num_z / 2 + 0.5) * fov_arg.pixel_l_z

                    if pos_x ** 2 + pos_y ** 2 < (fov_arg.diameter_roi / 2) ** 2:
                        idx_k = idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + idx_x
                        CRC_list.append(LIR[idx_k, idx_k].item())
                        Var_list.append(Cov[idx_k, idx_k].item())

        CRC_list = torch.tensor(CRC_list)
        Var_list = torch.tensor(Var_list)
        CRC_mean[i] = CRC_list.mean()
        Var_mean[i] = Var_list.mean()

        # eval points
        for j, idx_pt in enumerate(eval_arg.eval_point_idx):
            CRC_points[i, j] = LIR[:, idx_pt][idx_pt].item()
            Var_points[i, j] = Cov[:, idx_pt][idx_pt].item()

    if Var_points.size(1) > 0:
        save_path_tmp = f"{save_path}/Var_point_{name_val}"
        with open(save_path_tmp, "w") as file:
            Var_points.transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

        save_path_tmp = f"{save_path}/CRC_point_{name_val}"
        with open(save_path_tmp, "w") as file:
            CRC_points.transpose(0, 1).cpu().numpy().astype('float32').tofile(file)

    save_path_tmp = f"{save_path}/Var_mean_{name_val}"
    with open(save_path_tmp, "w") as file:
        Var_mean.cpu().numpy().astype('float32').tofile(file)

    save_path_tmp = f"{save_path}/CRC_mean_{name_val}"
    with open(save_path_tmp, "w") as file:
        CRC_mean.cpu().numpy().astype('float32').tofile(file)