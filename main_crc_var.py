import os
import torch
import numpy as np
import time
import crlb
import argparse

if __name__ == '__main__':
    with torch.no_grad():
        # get what kind of crc-var
        # [0]: sc; [1]: compton; [2]jscc
        flag_get = [1, 1, 1]

        # file path
        name_sys = "SysMat_20250307_JSCCGC_CollimatorT"
        name_factor_list = ["0mm", "3mm", "6mm"]

        # fov factors
        fov_arg = argparse.ArgumentParser().parse_args()
        fov_arg.pixel_num_x = 100
        fov_arg.pixel_num_y = 100
        fov_arg.pixel_num_z = 1
        fov_arg.pixel_num = fov_arg.pixel_num_x * fov_arg.pixel_num_y * fov_arg.pixel_num_z

        fov_arg.pixel_l_x = 3  # mm
        fov_arg.pixel_l_y = 3
        fov_arg.pixel_l_z = 3
        fov_arg.fov_z = -100
        fov_arg.diameter_roi = 300

        # energy factors (for compton)
        compton_arg = argparse.ArgumentParser().parse_args()
        compton_arg.e0 = 0.662  # energy of incident photons
        ene_resolution_662keV = 0.1  # energy resolution at 662keV
        compton_arg.ene_resolution = ene_resolution_662keV * (0.662 / compton_arg.e0) ** 0.5
        compton_arg.ene_threshold_max = 0.477
        compton_arg.ene_threshold_min = 0.050

        compton_arg.delta_r1 = 1.25      # mm
        compton_arg.delta_r2 = 1.25      # mm
        compton_arg.photon_num = 1e9

        compton_arg.num_workers = 50
        compton_arg.ds = 1      # down sampling ratio of list data

        # beta level
        lvl_start = -9
        lvl_end = -1

        # eval points
        eval_arg = argparse.ArgumentParser().parse_args()
        eval_point_x = torch.tensor([], dtype=torch.float32)
        eval_point_y = torch.tensor([], dtype=torch.float32)
        eval_point_z = torch.tensor([], dtype=torch.float32)

        # save path
        save_path = "./Result"

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
        img = torch.ones((fov_arg.pixel_num, 1), dtype=torch.float32).to(device, non_blocking=True)

        # generate beta
        eval_arg.eval_beta = []
        for lvl in range(lvl_start, lvl_end):
            eval_arg.eval_beta.append(1.0 * 10 ** lvl)
            eval_arg.eval_beta.append(3.0 * 10 ** lvl)

        # generate eval point index
        eval_arg.eval_point_idx = []
        if eval_point_x.size(0) > 0:
            idx_x = torch.floor(eval_point_x / fov_arg.pixel_l_x + fov_arg.pixel_num_x / 2).int()
            idx_y = torch.floor(eval_point_y / fov_arg.pixel_l_y + fov_arg.pixel_num_y / 2).int()
            idx_z = torch.floor(eval_point_z / fov_arg.pixel_l_z + fov_arg.pixel_num_z / 2).int()
            eval_arg.eval_point_idx.append(idx_z * fov_arg.pixel_num_x * fov_arg.pixel_num_y + idx_y * fov_arg.pixel_num_x + idx_x)

        print("Checking Devices ends, time used:", time.time() - time_start, "s")

        # --------Step2: Construct R Matrix--------
        print("--------Step2: Construct R Matrix--------")
        print("Construct R Matrix starts")

        R = crlb.get_r_matrix(fov_arg).to(device, non_blocking=True)
        print("Construct R Matrix ends, time used:", time.time() - time_start, "s")

        # --------Step3: Main Iteration--------
        print("--------Step3: Main Iteration--------")
        print("Main Iteration starts")
        crlb.get_crc_var(flag_get, name_factor_list, R, img, eval_arg, fov_arg, compton_arg, save_path, name_sys, device)

        print("Total time used:", time.time() - time_start)