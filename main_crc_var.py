import os
import torch
import numpy as np
import time

if __name__ == '__main__':
    with torch.no_grad():
        # file path
        name_sys = "SysMat_20250307_JSCCGC_CollimatorT"
        name_factor_list = ["0mm", "3mm"]

        # fov factors
        pixel_num_x = 100
        pixel_num_y = 100
        pixel_num_z = 1
        pixel_num = pixel_num_x * pixel_num_y * pixel_num_z

        pixel_l_x = 3  # mm
        pixel_l_y = 3
        pixel_l_z = 3
        diameter_roi = 300

        # beta level
        lvl_start = -9
        lvl_end = -1

        # eval points
        eval_point_x = torch.tensor([], dtype=torch.float32)
        eval_point_y = torch.tensor([], dtype=torch.float32)
        eval_point_z = torch.tensor([], dtype=torch.float32)

        # save path
        save_path = f"./Result/{name_sys}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # --------Step1: Checking Devices--------
        time_start = time.time()
        print("")
        print("--------Step1: Checking Devices--------")
        print("Checking Devices starts")
        # judge if CUDA is available and set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available, running on GPU")
        else:
            device = torch.device("cpu")
            print("CUDA is not available, running on CPU")

        # calculate constant variables
        # img to calculate
        img = torch.ones((pixel_num, 1), dtype=torch.float32).to(device, non_blocking=True)

        # generate beta
        eval_beta = []
        for lvl in range(lvl_start, lvl_end):
            eval_beta.append(1.0 * 10 ** lvl)
            eval_beta.append(3.0 * 10 ** lvl)

        # generate eval point index
        eval_point_idx = []
        if eval_point_x.size(0) > 0:
            idx_x = torch.floor(eval_point_x / pixel_l_x + pixel_num_x / 2).int()
            idx_y = torch.floor(eval_point_y / pixel_l_y + pixel_num_y / 2).int()
            idx_z = torch.floor(eval_point_z / pixel_l_z + pixel_num_z / 2).int()
            eval_point_idx.append(idx_z * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + idx_x)

        print("Checking Devices ends, time used:", time.time() - time_start, "s")


        # --------Step2: Construct R Matrix--------
        print("--------Step2: Construct R Matrix--------")
        print("Construct R Matrix starts")

        R = torch.zeros((pixel_num, pixel_num), dtype=torch.float32)
        for idx_x in range(pixel_num_x):
            for idx_y in range(pixel_num_y):
                for idx_z in range(pixel_num_z):
                    idx_k = idx_z * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + idx_x
                    # self
                    R[idx_k, idx_k] = 4

                    # neighbour of x
                    if idx_x > 0 and idx_x + 1 < pixel_num_x:
                        idx_l_1 = idx_z * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + (idx_x + 1)
                        idx_l_2 = idx_z * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + (idx_x - 1)
                        R[idx_k, idx_l_1] = -1
                        R[idx_k, idx_l_2] = -1

                    # neighbour of y
                    if idx_y > 0 and idx_y + 1 < pixel_num_y:
                        idx_l_1 = idx_z * pixel_num_x * pixel_num_y + (idx_y + 1) * pixel_num_x + idx_x
                        idx_l_2 = idx_z * pixel_num_x * pixel_num_y + (idx_y - 1) * pixel_num_x + idx_x
                        R[idx_k, idx_l_1] = -1
                        R[idx_k, idx_l_2] = -1

                    if idx_z > 0 and idx_z + 1 < pixel_num_z:
                        idx_l_1 = (idx_z + 1) * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + idx_x
                        idx_l_2 = (idx_z - 1) * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + idx_x
                        R[idx_k, idx_l_1] = -1
                        R[idx_k, idx_l_2] = -1

        R = R.to(device, non_blocking=True)
        print("Construct R Matrix ends, time used:", time.time() - time_start, "s")

        # --------Step3: Main Iteration--------
        print("--------Step3: Main Iteration--------")
        print("Main Iteration starts")
        for name_val in name_factor_list:
            # load sysmat
            print(name_val, "starts")
            sysmat_file_path = "./SysMat/" + name_sys + "/" + name_val
            sysmat = torch.from_numpy(np.reshape(np.fromfile(sysmat_file_path, dtype=np.float32), [pixel_num, -1])).transpose(0, 1)
            sysmat = sysmat.to(device, non_blocking=True)

            # Calculate average sensitivity（uniform phantom over whole fov）
            avg_sens = torch.sum(torch.matmul(sysmat, img)).item() / pixel_num
            print(f"Average Sensitivity = {avg_sens:.6f}")

            # Calculate FIM
            proj = torch.matmul(sysmat, img)
            proj_inv = proj.reciprocal()
            tmp_sysmat = sysmat * proj_inv
            FIM = torch.matmul(sysmat.t(), tmp_sysmat)

            # results container
            num_levels = len(eval_beta)
            CRC_points = torch.zeros((num_levels, len(eval_point_idx)), dtype=torch.float32)
            Var_points = torch.zeros_like(CRC_points)

            CRC_mean = torch.zeros(num_levels, dtype=torch.float32)
            CRC_std = torch.zeros_like(CRC_mean)
            Var_mean = torch.zeros_like(CRC_mean)
            Var_std = torch.zeros_like(CRC_mean)

            for i, beta in enumerate(eval_beta):
                # calculate LIR COV
                A = FIM + beta * R
                FIM_inv = torch.inverse(A)
                LIR = FIM_inv.matmul(FIM)
                Cov = LIR.matmul(FIM_inv)

                # ROI 内部像素统计
                CRC_list = []
                Var_list = []

                for idx_x in range(pixel_num_x):
                    for idx_y in range(pixel_num_y):
                        for idx_z in range(pixel_num_z):
                            pos_x = (idx_x - pixel_num_x / 2 + 0.5) * pixel_l_x
                            pos_y = (idx_y - pixel_num_y / 2 + 0.5) * pixel_l_y
                            pos_z = (idx_z - pixel_num_z / 2 + 0.5) * pixel_l_z

                            if pos_x**2 + pos_y**2 < (diameter_roi/2)**2:
                                idx_k = idx_z * pixel_num_x * pixel_num_y + idx_y * pixel_num_x + idx_x
                                CRC_list.append(LIR[idx_k, idx_k].item())
                                Var_list.append(Cov[idx_k, idx_k].item())

                CRC_list = torch.tensor(CRC_list)
                Var_list = torch.tensor(Var_list)
                CRC_mean[i] = CRC_list.mean()
                Var_mean[i] = Var_list.mean()

                # eval points
                for j, idx_pt in enumerate(eval_point_idx):
                    CRC_points[i, j] = LIR[:, idx_pt][idx_pt].item()
                    Var_points[i, j] = Cov[:, idx_pt][idx_pt].item()

            # save
            if eval_point_x.size(0) > 0:
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

            print(name_val, "ends, time used:", time.time() - time_start, "s")
