import main
import json
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from compute_f import split_ts_seq, compute_step_positions, compute_steps, compute_headings, compute_stride_length, compute_step_heading, compute_rel_positions
from io_f import read_data_file

from visualize_f import visualize_trajectory, visualize_heatmap, save_figure_to_html
from main import calibrate_magnetic_wifi_ibeacon_to_position

# for data preprocessing. The data has the quality issue that the deliminater misses in some lines
VALUES_IN_LINE = {
    "TYPE_WAYPOINT": 4,
    "TYPE_ACCELEROMETER": 5,
    "TYPE_MAGNETIC_FIELD": 5,
    "TYPE_GYROSCOPE": 5,
    "TYPE_ROTATION_VECTOR": 5,
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED": 5,
    "TYPE_GYROSCOPE_UNCALIBRATED": 5,
    "TYPE_ACCELEROMETER_UNCALIBRATED": 5,
    "TYPE_WIFI": 7,
    "TYPE_BEACON": 10
}

TABLE_OF_FLOOR = {
    "B3": -3,
    "3B": -3,
    "B2": -2,
    "2B": -2,
    "B1": -1,
    "1B": -1,
    "F1": 1,
    "1F": 1,
    "F2": 2,
    "2F": 2,
    "F3": 3,
    "3F": 3,
    "F4": 4,
    "4F": 4,
    "F5": 5,
    "5F": 5,
    "F6": 6,
    "6F": 6,
    "F7": 7,
    "7F": 7,
    "F8": 8,
    "8F": 8,
    "F9": 9,
    "9F": 9,
    "F10": 10,
    "10F": 10,
}

SUBMISSION_SITES = {
    '5a0546857ecc773753327266',
    '5c3c44b80379370013e0fd2b',
    '5d27075f03f801723c2e360f',
    '5d27096c03f801723c31e5e0',
    '5d27097f03f801723c320d97',
    '5d27099f03f801723c32511d',
    '5d2709a003f801723c3251bf',
    '5d2709b303f801723c327472',
    '5d2709bb03f801723c32852c',
    '5d2709c303f801723c3299ee',
    '5d2709d403f801723c32bd39',
    '5d2709e003f801723c32d896',
    '5da138274db8ce0c98bbd3d2',
    '5da1382d4db8ce0c98bbe92e',
    '5da138314db8ce0c98bbf3a0',
    '5da138364db8ce0c98bc00f1',
    '5da1383b4db8ce0c98bc11ab',
    '5da138754db8ce0c98bca82f',
    '5da138764db8ce0c98bcaa46',
    '5da1389e4db8ce0c98bd0547',
    '5da138b74db8ce0c98bd4774',
    '5da958dd46f8266d0737457b',
    '5dbc1d84c1eb61796cf7c010',
    '5dc8cea7659e181adb076a3f'
}

def create_magnetic_features(paths:list, floorNo:str, site:str):
    '''
        params:
            paths:
                a set of strings of path. They are paths in the same floor and site.
            floorNo:
                the string of same number of selected paths.
                ex) F3, F2, B1
            site:


    '''
    # declaration
    wifi_features = pd.DataFrame()
    for i, path in tqdm(enumerate(paths)):
        data = calibrate_magnetic_wifi_ibeacon_to_position([path])
        features = process_magnetic_feature(data)
        waypoints = create_interpolate_waypoint(data, floorNo, site)
        if i == 0: wifi_features = pd.concat([waypoints, features], axis=1)
        else: 
            wifi_features = wifi_features.append(pd.concat([waypoints, features], axis=1))

    return wifi_features


def create_interpolate_waypoint(mwi, floorNo, site):
    df_gt = pd.DataFrame(mwi.keys(), columns=["x","y"])
    df_gt = pd.concat([df_gt,pd.Series(name="floor", data=[convert_floor_to_int(floorNo) for _ in range(df_gt.shape[0])])], ignore_index=True, axis=1)
    df_gt = pd.concat([df_gt,pd.Series(name="floor", data=[site for _ in range(df_gt.shape[0])])], ignore_index=True, axis=1)
    df_gt.columns = ["x", "y", "floor", "site"]
    return df_gt


def process_magnetic_feature(mwi):
    mag_feature_df = []
    for position_key in mwi:
        magnetic_data = mwi[position_key]['magnetic']
        magnetic_s = np.mean(np.sqrt(np.sum(magnetic_data[:, 1:4] ** 2, axis=1)))
        magnetic_1_mean = np.mean(magnetic_data[:, 1])
        magnetic_2_mean = np.mean(magnetic_data[:, 2])
        magnetic_3_mean = np.mean(magnetic_data[:, 3])
        magnetic_1_std = np.std(magnetic_data[:, 1])
        magnetic_2_std = np.std(magnetic_data[:, 2])
        magnetic_3_std = np.std(magnetic_data[:, 3])
        magnetic_cov = np.cov(magnetic_data.T)
        tmp_data = [
            magnetic_s,
            magnetic_1_mean,
            magnetic_2_mean,
            magnetic_3_mean,
            magnetic_1_std,
            magnetic_2_std,
            magnetic_3_std,
        ]
        for v in magnetic_cov.reshape(-1): tmp_data.append(v)
        mag_feature_df.append(tmp_data)
    mag_feature_df = pd.DataFrame(mag_feature_df)
    return mag_feature_df


def concatenate_formatting_waypoints(paths):
    rel = []
    for path in tqdm(paths):
        rel.extend(formatting_for_visualize_waypoint(path))
    return np.array(rel)


def convert_floor_to_int(floor:str):
    return TABLE_OF_FLOOR[floor] if floor in TABLE_OF_FLOOR.keys() else False

def formatting_for_visualize_waypoint(path):
    data = read_data_file(path)
    trajectory = data.waypoint[:,1:3]
    return trajectory

def scrape_building_name(path):
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line_data = line.split()
        if not line_data:
            continue
        if (line_data[0] == '#') & (line_data[1].startswith('SiteID:')):
            site = line_data[1].split("SiteID:")[1]
            return site
    return false

def split_sample_sbm_to_dataframe(smpl_sbm_path="./sample_submission.csv"):
    site, path, timestamp = [],[],[]
    for line in pd.read_csv(smpl_sbm_path)["site_path_timestamp"]:
        s,p,t = line.split("_")
        site.append(s)
        path.append(p)
        timestamp.append(t)
    required_pred_data = pd.DataFrame([site,path,timestamp]).T
    required_pred_data.columns = ["site","path","time"]
    return required_pred_data


def uncalibrate_magnetic_wifi_ibeacon_to_position(path_file_list):
    mwi_datas = {}
    
    for path_filename in path_file_list:
        print(f'Processing {path_filename}...')

        path_datas = read_data_file(path_filename)
        wifi_features = process_magnetic_feature(path_datas)
        acce_datas = path_datas.acce
        magn_datas = path_datas.magn
        ahrs_datas = path_datas.ahrs
        wifi_datas = path_datas.wifi
        ibeacon_datas = path_datas.ibeacon

        step_positions = compute_step_positions_without_cb(acce_datas, ahrs_datas)
        # visualize_trajectory(posi_datas[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Ground Truth', show=True)
        # visualize_trajectory(step_positions[:, 1:3], floor_plan_filename, width_meter, height_meter, title='Step Position', show=True)

        if wifi_datas.size != 0:
            sep_tss = np.unique(wifi_datas[:, 0].astype(float))
            wifi_datas_list = split_ts_seq(wifi_datas, sep_tss)
            for wifi_ds in wifi_datas_list:
                diff = np.abs(step_positions[:, 0] - float(wifi_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['wifi'] = np.append(mwi_datas[target_xy_key]['wifi'], wifi_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': wifi_ds,
                        'ibeacon': np.zeros((0, 3))
                    }

        if ibeacon_datas.size != 0:
            sep_tss = np.unique(ibeacon_datas[:, 0].astype(float))
            ibeacon_datas_list = split_ts_seq(ibeacon_datas, sep_tss)
            for ibeacon_ds in ibeacon_datas_list:
                diff = np.abs(step_positions[:, 0] - float(ibeacon_ds[0, 0]))
                index = np.argmin(diff)
                target_xy_key = tuple(step_positions[index, 1:3])
                if target_xy_key in mwi_datas:
                    mwi_datas[target_xy_key]['ibeacon'] = np.append(mwi_datas[target_xy_key]['ibeacon'], ibeacon_ds, axis=0)
                else:
                    mwi_datas[target_xy_key] = {
                        'magnetic': np.zeros((0, 4)),
                        'wifi': np.zeros((0, 5)),
                        'ibeacon': ibeacon_ds
                    }

        sep_tss = np.unique(magn_datas[:, 0].astype(float))
        magn_datas_list = split_ts_seq(magn_datas, sep_tss)
        for magn_ds in magn_datas_list:
            diff = np.abs(step_positions[:, 0] - float(magn_ds[0, 0]))
            index = np.argmin(diff)
            target_xy_key = tuple(step_positions[index, 1:3])
            if target_xy_key in mwi_datas:
                mwi_datas[target_xy_key]['magnetic'] = np.append(mwi_datas[target_xy_key]['magnetic'], magn_ds, axis=0)
            else:
                mwi_datas[target_xy_key] = {
                    'magnetic': magn_ds,
                    'wifi': np.zeros((0, 5)),
                    'ibeacon': np.zeros((0, 3))
                }

    return mwi_datas

def compute_step_positions_without_cb(acce_datas, ahrs_datas):
    step_timestamps, step_indexs, step_acce_max_mins = compute_steps(acce_datas)
    headings = compute_headings(ahrs_datas)
    stride_lengths = compute_stride_length(step_acce_max_mins)
    step_headings = compute_step_heading(step_timestamps, headings)
    rel_positions = compute_rel_positions(stride_lengths, step_headings)
    # step_positions = correct_positions(rel_positions, posi_datas)

    return rel_positions
