import os
import torch
import numpy as np
import time
from process_list import get_compton_backproj_list

def get_coor_plane(fov_arg):
    # get the coordinate of fov
    fov_coor = torch.ones([fov_arg.pixel_num_x, fov_arg.pixel_num_y, 3])
    min_x = -(fov_arg.pixel_num_x / 2 - 0.5) * fov_arg.pixel_l_x
    max_x = (fov_arg.pixel_num_x / 2 - 0.5) * fov_arg.pixel_l_x
    min_y = -(fov_arg.pixel_num_y / 2 - 0.5) * fov_arg.pixel_l_y
    max_y = (fov_arg.pixel_num_y / 2 - 0.5) * fov_arg.pixel_l_y
    fov_coor[:, :, 0] *= torch.linspace(min_x, max_x, fov_arg.pixel_num_x).reshape([1, -1])
    fov_coor[:, :, 1] *= torch.linspace(min_y, max_y, fov_arg.pixel_num_y).reshape([-1, 1])
    fov_coor[:, :, 2] = fov_arg.fov_z
    fov_coor = fov_coor.reshape(-1, 3)
    return fov_coor

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

def get_sysmat_sc(fov_arg, name_sys, name_val, device):
    # get sysmat of sc
    sysmat_file_path = "./SysMat/" + name_sys + "/SC/" + name_val
    sysmat_sc = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32),[fov_arg.pixel_num, -1])).transpose(0, 1)
    sysmat_sc = sysmat_sc.to(device, non_blocking=True)
    return sysmat_sc

def get_sysmat_compton(fov_arg, compton_arg, name_sys, name_val, device):
    # get sysmat of compton
    time_start = time.time()

    # get sysmat of sc
    sysmat_file_path = "./SysMat/" + name_sys + "/SC/" + name_val
    sysmat_sc = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32),[fov_arg.pixel_num, -1])).transpose(0, 1)
    sysmat_sc = sysmat_sc.to(device, non_blocking=True)

    # load list data
    list_file_path = "./SysMat/" + name_sys + "/Compton/" + name_val + ".csv"
    list_origin = torch.from_numpy(np.genfromtxt(list_file_path, delimiter=",", dtype=np.float32)[:, 0:4])
    list_origin = list_origin[0:int(list_origin.size(0) * compton_arg.ds), :]   # down sampling

    # load detector
    detector_file_path = "./SysMat/" + name_sys + "/Detector/" + name_val + ".csv"
    detector = torch.from_numpy(np.genfromtxt(detector_file_path, delimiter=",", dtype=np.float32)[:, 1:4])
    detector = detector.to(device, non_blocking=True)

    # get fov coor
    coor_plane = get_coor_plane(fov_arg).to(device, non_blocking=True)

    # process list
    sysmat_compton = []
    list_origin_chunks = torch.chunk(list_origin, compton_arg.num_workers, dim=0)
    for list_origin_tmp_chunk in list_origin_chunks:
        sysmat_compton_chunk = get_compton_backproj_list(list_origin_tmp_chunk.to(device), compton_arg, detector, coor_plane, sysmat_sc, device)
        sysmat_compton.append(sysmat_compton_chunk)
        torch.cuda.empty_cache()
        print("Chunk Num", str(len(sysmat_compton)), "ends, time used:", time.time() - time_start, "s")

    sysmat_compton = torch.cat(sysmat_compton, dim=0)
    avg_sens_tmp = torch.sum(sysmat_compton).item() / fov_arg.pixel_num
    avg_sens_true = sysmat_compton.size(0) / compton_arg.photon_num
    sysmat_compton = sysmat_compton * avg_sens_true / avg_sens_tmp
    sysmat_compton = sysmat_compton.to(device, non_blocking=True)

    return sysmat_compton

def get_crc_var_single(sysmat, R, img, eval_arg, fov_arg, save_path, name_val):
    print("----", name_val, "starts----")

    # makedirs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

def get_crc_var(flag_get, name_factor_list, R, img, eval_arg, fov_arg, compton_arg, save_path, name_sys, device):
    time_start = time.time()
    for name_val in name_factor_list:
        # get sysmat
        if sum(flag_get) > 1 or flag_get[2] == 1:
            sysmat_sc = get_sysmat_sc(fov_arg, name_sys, name_val, device)
            sysmat_compton = get_sysmat_compton(fov_arg, compton_arg, name_sys, name_val, device)
        elif flag_get[1] == 1:
            sysmat_compton = get_sysmat_compton(fov_arg, compton_arg, name_sys, name_val, device)
        else:
            sysmat_sc = get_sysmat_sc(fov_arg, name_sys, name_val, device)

        # get and save crc-var
        if flag_get[0] == 1:
            save_path_tmp_sc = f"{save_path}/{name_sys}/SC"
            get_crc_var_single(sysmat_sc, R, img, eval_arg, fov_arg, save_path_tmp_sc, name_val)

        if flag_get[1] == 1:
            save_path_tmp_compton = f"{save_path}/{name_sys}/Compton"
            get_crc_var_single(sysmat_compton, R, img, eval_arg, fov_arg, save_path_tmp_compton, name_val)

        if flag_get[2] == 1:
            sysmat_jscc = torch.cat((sysmat_sc, sysmat_compton), dim=0)
            save_path_tmp_jscc = f"{save_path}/{name_sys}/JSCC"
            get_crc_var_single(sysmat_jscc, R, img, eval_arg, fov_arg, save_path_tmp_jscc, name_val)

        print(name_val, "ends, time used:", time.time() - time_start, "s")