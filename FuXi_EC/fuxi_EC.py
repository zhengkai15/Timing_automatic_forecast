# -*-coding:utf-8 -*-
"""
# File       : submit2oss.py
# Time       ：12/29/23 10:59 AM
# Author     ：KaiZheng
# version    ：python 3.8
# Project  : Fuxi-infer-online
# Description：
"""

import argparse
import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from util import save_like
from copy import deepcopy
from make_hres_input import make_hers_input_merge, ExecutionTimer

flag = './FuXi_EC'
ort.set_default_logger_severity(3)

timer = ExecutionTimer()

parser = argparse.ArgumentParser(description="Process EC data.")
parser.add_argument('--model', type=str, required=False, default=f'{flag}/model/', help="FuXi onnx model dir")
parser.add_argument("--tpsetzero", default=True, type=bool, help="Zero field data is no precipitation, at this time, the parameter can\
                                                                    be ignored, if the download EC for the forecast moment,\
                                                                    need to zero precipitation, the parameter needs to be set to True")
parser.add_argument("--init_time", default="20231012", help="Initial time")
parser.add_argument("--raw_data_root_path", default=f'{flag}/input_meta',
                    help="Root path for raw data")
parser.add_argument("--init_time_type", default="case1", help="Type of initial time")
parser.add_argument('--input_save_dir', type=str, default=f'{flag}/input')
parser.add_argument("--save_dir", default=f'{flag}/output', help="Directory for saving data")
parser.add_argument("--raw_data_format_type", default="netcdf", help="Raw data dataset type")
parser.add_argument('--num_steps', type=int, nargs="+", default=[20])
parser.add_argument('--device', type=str, default="cuda", help="The device to run FuXi model")

args = parser.parse_args()
# stages = ['short', 'medium']
stages = ['short']

def time_encoding(init_time, total_step, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(total_step):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = np.array(times, dtype=np.float32)
        temb = np.concatenate([np.sin(temb), np.cos(temb)], axis=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)


def load_model(model_name, device):
    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption

    if device == "cuda":
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
    elif device == "cpu":
        providers=['CPUExecutionProvider']
        options.intra_op_num_threads = 24
    else:
        raise ValueError("device must be cpu or cuda!")

    session = ort.InferenceSession(
        model_name,
        sess_options=options,
        providers=providers
    )
    return session


def run_inference(models, input, total_step, save_dir=""):
    hist_time = pd.to_datetime(input.time.values[-2])
    init_time = pd.to_datetime(input.time.values[-1])
    assert init_time - hist_time == pd.Timedelta(hours=6)

    lat = input.lat.values
    lon = input.lon.values
    assert lat[0] == 90 and lat[-1] == -90
    batch = input.values[None]
    print(f'Model initial Time: {init_time.strftime(("%Y%m%d%H"))}')
    print(f"Region: {lat[0]:.2f} ~ {lat[-1]:.2f}, {lon[0]:.2f} ~ {lon[-1]:.2f}")

    print(f'Inference ...')
    start = time.perf_counter()
    for step in range(total_step):
        lead_time = (step + 1) * 6
        valid_time = init_time + pd.Timedelta(hours=step * 6)

        stage = stages[min(len(models) - 1, step // 20)]
        model = models[stage]

        input_names = [x.name for x in model.get_inputs()]
        inputs = {'input': batch}

        if "step" in input_names:
            inputs['step'] = np.array([step], dtype=np.float32)

        if "hour" in input_names:
            hour = valid_time.hour / 24
            inputs['hour'] = np.array([hour], dtype=np.float32)

        t0 = time.perf_counter()
        new_input, = model.run(None, inputs)
        output = deepcopy(new_input[:, -1:])
        step_time = time.perf_counter() - t0
        print(f"stage: {stage}, lead_time: {lead_time:03d} h, step_time: {step_time:.3f} sec")

        save_like(output, input, lead_time, save_dir)
        batch = new_input

    run_time = time.perf_counter() - start
    print(f'Inference done take {run_time:.2f}')


if __name__ == "__main__":

    init_time_type_dict = {"case1": (0, 6), "case2": (6, 12), "case3": (12, 18), "case4": (18, 0)}

    # Build file paths
    if args.raw_data_format_type == "grib":
        file_suffix = "grib"
    if args.raw_data_format_type == "netcdf":
        file_suffix = "nc"

    file_hist_sfc = f"{args.raw_data_root_path}/sfc/{init_time_type_dict[args.init_time_type][0]:02d}/{args.init_time}.{file_suffix}"
    file_hist_pl = f"{args.raw_data_root_path}/pl/{init_time_type_dict[args.init_time_type][0]:02d}/{args.init_time}.{file_suffix}"
    if args.init_time_type == "case4":
        init_time_case4 = (pd.to_datetime(args.init_time) + pd.Timedelta('1D')).strftime("%Y%m%d")
        file_init_sfc = f"{args.raw_data_root_path}/sfc/{init_time_type_dict[args.init_time_type][1]:02d}/{init_time_case4}.{file_suffix}"
        file_init_pl = f"{args.raw_data_root_path}/pl/{init_time_type_dict[args.init_time_type][1]:02d}/{init_time_case4}.{file_suffix}"
    else:
        file_init_sfc = f"{args.raw_data_root_path}/sfc/{init_time_type_dict[args.init_time_type][1]:02d}/{args.init_time}.{file_suffix}"
        file_init_pl = f"{args.raw_data_root_path}/pl/{init_time_type_dict[args.init_time_type][1]:02d}/{args.init_time}.{file_suffix}"
    print(file_hist_sfc)
    data, save_name = make_hers_input_merge(file_hist_sfc, file_hist_pl, file_init_sfc, file_init_pl,
                                            args.input_save_dir, args.tpsetzero, args.raw_data_format_type)

    print("data.level.values:", data.level.values)

    save_dir = os.path.join(args.save_dir, save_name.split('.nc')[-2] + "-output")
    print(save_dir)
    models = {}
    for stage in stages:
        model_path = os.path.join(args.model, f"{stage}.onnx")
        if os.path.exists(model_path):
            start = time.perf_counter()
            print(f'Load FuXi {stage} ...')
            model = load_model(model_path, args.device)
            models[stage] = model
            print(f'Load FuXi {stage} take {time.perf_counter() - start:.2f} sec')

    run_inference(models, data, args.num_steps[0], save_dir)

    print("\033[92mCongratulations. Your inference is complete!\033[0m")
