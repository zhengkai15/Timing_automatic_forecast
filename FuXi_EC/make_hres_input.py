# -*-coding:utf-8 -*-
"""
# File       : submit2oss.py
# Time       ：12/29/23 10:59 AM
# Author     ：KaiZheng
# version    ：python 3.8
# Project  : Fuxi-infer-online
# Description：
"""
import os
import numpy as np
import pandas as pd
import pygrib as pg
import xarray as xr

def print_level_info(ds):
    # Assuming 'ds' is the object you want to check
    if isinstance(ds, xr.DataArray):
        print("The object is a DataArray")
    elif isinstance(ds, xr.Dataset):
        ds = ds.to_array()
        print("The object is a DataSet")
    else:
        print("The object is neither a DataArray nor a DataSet")
    check_names = [
        'Z500', 'Z850',
        'T500', 'T850',
        'U500', 'U850',
        'V500', 'V850',
        'R500', 'R850',
        'T2M', 'U10', 'V10', 'MSL', 'TP'
    ]
    for lvl in ds.level.values:
        if lvl.upper() in check_names:
            v = ds.sel(level=lvl).values
            print(f'{lvl}: {v.shape}, {v.min():.3f} ~ {v.max():.3f}')


import time as tm


class ExecutionTimer:
    def __init__(self):
        self.last_time = tm.perf_counter()

    def format_time(self, seconds):
        """将秒数转换为时分秒和毫秒格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours} hours, {minutes} minutes, {int(seconds)} seconds and {milliseconds} milliseconds"

    def print_elapsed_time(self, message=""):
        """打印从上次记录时间到当前的经过时间，并更新时间戳"""
        current_time = tm.perf_counter()
        execution_time = current_time - self.last_time
        formatted_time = self.format_time(execution_time)
        print(f"{message} - Time elapsed: {formatted_time}.")
        self.last_time = current_time


def transform2_lon(data, lon):
    idata = data.copy()
    idata[:, :720], idata[:, 720:] = data[:, 720:], data[:, :720]
    ilon = lon.copy()
    ilon[:720], ilon[720:] = lon[720:], lon[:720] + 360
    return idata, ilon


# def calculate_relative_humidity(temperature, absolute_humidity, pressure):
#     # temperature代表绝对温度（单位：开尔文），absolute_humidity代表绝对湿度（单位：克/立方米），pressure代表气压（单位：帕斯卡）。函数将返回相对湿度（以百分比表示）
#     # 首先，将温度转换为摄氏度
#     temperature_celsius = temperature - 273.15
#
#     # 计算饱和水汽压力（根据温度）
#     saturated_vapor_pressure = 6.112 * 10 ** ((7.5 * temperature_celsius) / (237.7 + temperature_celsius))
#
#     # 计算实际水汽压力
#     actual_vapor_pressure = absolute_humidity * pressure / 0.622
#
#     # 计算相对湿度（以百分比表示）
#     relative_humidity = (actual_vapor_pressure / saturated_vapor_pressure) * 100
#
#     return relative_humidity

def calculate_relative_humidity(temperature, specific_humidity, pressure):
    # temperature代表温度（单位：开尔文），specific_humidity代表绝对湿度（单位：千克/千克），pressure代表气压（单位：帕斯卡）。函数将返回相对湿度（以百分比表示）
    # 首先，将温度转换为摄氏度
    temperature_celsius = temperature - 273.15

    # 计算饱和水汽压力（根据温度）
    saturated_vapor_pressure = 6.112 * 10 ** ((17.67 * temperature_celsius) / (temperature_celsius + 243.5))

    # 计算实际水汽压力
    actual_vapor_pressure = specific_humidity * pressure / (0.622 + 0.378 * specific_humidity)

    # 计算相对湿度（以百分比表示）
    relative_humidity = (actual_vapor_pressure / saturated_vapor_pressure) * 100
    return relative_humidity


def shum_switch_rhum(temp, shum, pres):
    '''
    https://blog.csdn.net/Soul_taker/article/details/126534580
    利用比湿（specific humidity）计算相对湿度
    :param temp: 气温，K
    :param shum: 比湿
    :param pres: 气压，Pa
    :return: rhum，%
    '''
    rhum = 0.236 * pres * shum * np.exp((17.67 * (temp - 273.16)) / (temp - 29.65)) ** (-1)
    # print(rhum)
    return rhum


def make_hres_input_zero_field_grib_format(sfc_path, pl_path, tpsetzero=True):
    print(f'sfc_path:{sfc_path}')
    print(os.getcwd())
    assert os.path.exists(sfc_path)
    assert os.path.exists(pl_path)

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ['z', 't', 'u', 'v', 'q', 'clwc']
    sf_names = ['2t', '2d', 'sst', '10u', '10v',
                '100u', '100v', 'msl', 'sp', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp']

    try:
        ds_pl = pg.open(pl_path)
        ds_sfc = pg.open(sfc_path)
    except:
        print(f"\033[92m Failed to open file!\033[0m")

    '''
    print("\033[92m***********pl数据信息************\033[0m")
    for i, item in enumerate(ds_pl,1):
        print("item.date-->:",item.date)
        print("item.time-->:",item.validDate)
        print("item.level-->:",item.level)
        print("item.step-->:",item.step)
        print("item.shortName-->:",item)
    print("\033[92m***********pl数据信息************\033[0m")

    print("***************************************************") 

    print("\033[92m***********sfc数据信息************\033[0m")
    for i, item in enumerate(ds_sfc,1):
        print("item.date-->:",item.date)
        print("item.time-->:",item.validDate)
        print("item.level-->:",item.level)
        print("item.step-->:",item.step)
        print("item.shortName-->:",item)
    print("\033[92m***********sfc数据信息************\033[0m")
    '''
    input = []
    level = []

    for name in pl_names + sf_names:
        print("name:", name)
        if name in pl_names:
            try:
                data = ds_pl.select(shortName=name, level=levels)

            except:
                print("\033[92m pl wrong,can't found!\033[0m")

            data = data[:len(levels)]

            if len(data) != len(levels):
                print("\033[92m pl wrong,level wrong!\033[0m")

            for level_ in levels:
                # print(level_)
                if name == 'r':
                    try:
                        data_level = ds_pl.select(shortName=name, level=[level_])
                    except:
                        data_level = ds_pl.select(shortName='q', level=[level_])
                        data_t = ds_pl.select(shortName='t', level=[level_])
                        print('r is not exit, read q and t to calcute rhum')
                else:
                    data_level = ds_pl.select(shortName=name, level=[level_])
                for v in data_level:
                    if data_level[0].step == 0:
                        print("v.date:", data_level[0].date)
                        print("v.time:", data_level[0].validDate)
                        init_time = data_level[0].validDate
                        lat = data_level[0].distinctLatitudes
                        lon = data_level[0].distinctLongitudes
                        img, _, _ = data_level[0].data()
                        if name == 'r':
                            # img = calculate_relative_humidity(data_t[0].data()[0], img, level_ * 100)
                            img = shum_switch_rhum(data_t[0].data()[0], img, level_ * 100)
                        img, lon = transform2_lon(img, lon)
                        input.append(img)
                        level.append(f'{name}{v.level}')
                        # if (f'{name}{v.level}')=="z500":
                        #     np.save("z500.npy",v.values)
                        print(f"{v.name}: {v.level}, {img.shape}, {img.min()} ~ {img.max()}")

        if name in sf_names:
            try:
                data_sfc = ds_sfc.select(shortName=name)
                print("len(data_sfc):", len(data_sfc))
            except:
                print(f"\033[92m sfc{name} wrong,can't found!\033[0m")

            name_map = {'2t': 't2m', '2d': 'd2m',
                        'sst': 'sst',
                        '10u': 'u10', '10v': 'v10',
                        '100u': 'u100', '100v': 'v100',
                        'msl': 'msl', 'sp': 'sp',
                        'ssr': 'ssr', 'ssrd': 'ssrd',
                        'fdir': 'fdir', 'ttr': 'ttr',
                        'tp': 'tp'}

            name = name_map[name]

            for v in data_sfc:
                if v.step == 0:
                    img, _, _ = v.data()
                    img, _ = transform2_lon(img, lon)
                    if name == "tp" and (tpsetzero == True):
                        tp = img * 0
                        input.append(tp)
                        level.append("tp")
                        print("***************go into***************")
                    else:
                        input.append(img)
                        level.append(name)
                    print(f"{v.name}: {img.shape}, {img.min()} ~ {img.max()}")

    input = np.stack(input)
    print("input.shape:", input.shape)
    assert input.shape[-3:] == (92, 721, 1440)
    assert input.max() < 1e10

    level = [lvl.upper() for lvl in level]
    times = [pd.to_datetime(init_time)]
    input = xr.DataArray(
        data=input[None],
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': times, 'level': level, 'lat': lat, 'lon': lon},
    )
    if np.isnan(input).sum() > 0:
        print("\033[92mField has nan value\033[0m")
    print("input:", input)
    return input


def make_hres_input_zero_field_netcdf_format(sfc_path, pl_path, tpsetzero=True):
    assert os.path.exists(sfc_path)
    assert os.path.exists(pl_path)

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ['z', 't', 'u', 'v', 'q', 'clwc']
    sf_names = ['2t', '2d', 'sst', '10u', '10v',
                '100u', '100v', 'msl', 'sp', 'ssr', 'ssrd', 'fdir', 'ttr', 'tp']

    try:
        ds_pl = xr.open_dataset(pl_path)
        ds_sfc = xr.open_dataset(sfc_path)
    except:
        print(f"\033[92m Failed to open file!\033[0m")

    input = []
    level = []

    for name in pl_names + sf_names:
        print("name:", name)
        if name in pl_names:
            try:
                data = ds_pl[name].sel(level=levels)
            except:
                print("\033[92m pl wrong,can't found!\033[0m")

            if len(data.level.values) != len(levels):
                print("\033[92m pl wrong,level wrong!\033[0m")

            for lvl in levels:
                print("data.time.values:", data.time.values)
                init_time = data.time
                lat = data.latitude
                lon = data.longitude

                img0 = data.sel(level=lvl).values
                img = np.squeeze(img0)
                input.append(img)
                level.append(f'{name}{lvl}')
                # if (f'{name}{lvl}')=="z500":
                #     np.save("z500_nc.npy",img)
                # print(f"{img0.name}: {img0.level}, {img.shape}, {img.min()} ~ {img.max()}")
        print("\033[92m ************\033[0m")
        if name in sf_names:
            try:
                data_sfc = ds_sfc[name].sel().values
                # print("len(data_sfc):",len(data_sfc))
            except:
                print(f"\033[92m sfc{name} wrong,can't found!\033[0m")

            # name_map = {'2t': 't2m', '10u': 'u10', '10v': 'v10','msl':'msl','tp':'tp'}
            # name = name_map[name]

            for v in data_sfc:
                img = v
                if name == "tp" and (tpsetzero == True):
                    tp = img * 0
                    input.append(tp)
                    level.append("tp")
                    print("\033[92m***************go into***************\033[0m")
                else:
                    input.append(img)
                    level.append(name)
                # print(f"{v.name}: {img.shape}, {img.min()} ~ {img.max()}")
    # print("\033[92m input:\033[0m",input)
    input = np.stack(input)
    print("input.shape:", input.shape)
    assert input.shape[-3:] == (92, 721, 1440)
    assert input.max() < 1e10

    lat = np.arange(90, -90.25, -0.25)
    lon = np.arange(0, 360, 0.25)

    level = [lvl.upper() for lvl in level]
    times = pd.to_datetime(init_time)
    input = xr.DataArray(
        data=input[None],
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': times, 'level': level, 'lat': lat, 'lon': lon},
    )
    if np.isnan(input).sum() > 0:
        print("\033[92mField has nan value\033[0m")
    print("input:", input)
    return input


def make_hers_input_merge(file_hist_sfc, file_hist_pl, file_init_sfc, file_init_pl, save_dir, tpsetzero,
                          raw_data_format_type):
    os.makedirs(save_dir, exist_ok=True)
    os.chmod(save_dir, 0o777)
    if raw_data_format_type == "grib":
        save_name = file_init_sfc.split('/')[-1][:-5] + '-' + file_init_sfc.split('/')[-2] + '_input_grib.nc'
        try:
            ds = xr.open_dataarray(os.path.join(save_dir, save_name))
            print('input exists')
            return ds, save_name
        except:
            d1 = make_hres_input_zero_field_grib_format(file_hist_sfc, file_hist_pl, tpsetzero=tpsetzero)
            d2 = make_hres_input_zero_field_grib_format(file_init_sfc, file_init_pl, tpsetzero=tpsetzero)
    if raw_data_format_type == "netcdf":
        d1 = make_hres_input_zero_field_netcdf_format(file_hist_sfc, file_hist_pl, tpsetzero=tpsetzero)
        d2 = make_hres_input_zero_field_netcdf_format(file_init_sfc, file_init_pl, tpsetzero=tpsetzero)

    if d1 is not None and d2 is not None:
        # print("Start saving two-step data")
        ds = xr.concat([d1, d2], 'time')
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
        ds = ds.astype(np.float32)
        save_name = pd.to_datetime(d2.time.values[0]).strftime(f"%Y%m%d-%H_input_{raw_data_format_type}.nc")
        ds.to_netcdf(os.path.join(save_dir, save_name))
        print("\033[92m Input data saved successfully\033[0m")
        print("input-ds:", ds)
        print("ds.level.values:", ds.level.values)
    return ds, save_name


if __name__ == "__main__":
    import argparse
    import xarray as xr

    timer = ExecutionTimer()

    parser = argparse.ArgumentParser(description="Process EC data.")
    parser.add_argument("--tpsetzero", default=True, type=bool, help="Zero field data is no precipitation, at this time, the parameter can\
                                                                      be ignored, if the download EC for the forecast moment,\
                                                                      need to zero precipitation, the parameter needs to be set to True")
    parser.add_argument("--init_time", default="20231012", help="Initial time")
    parser.add_argument("--raw_data_root_path", default="./Sample_Data/hres_input_grib_raw/hres",
                        help="Root path for raw data")
    parser.add_argument("--init_time_type", default="case1", help="Type of initial time")
    parser.add_argument("--save_dir", default="./", help="Directory for saving data")
    parser.add_argument("--raw_data_format_type", default="grib", help="Raw data dataset type")

    args = parser.parse_args()

    init_time_type_dict = {"case1": (0, 6), "case2": (6, 12), "case3": (12, 18), "case4": (18, 0)}

    # Build file paths
    if args.raw_data_format_type == "grib":
        file_suffix = "grib"
    if args.raw_data_format_type == "netcdf":
        file_suffix = "nc"

    file_hist_sfc = f"{args.raw_data_root_path}/sfc/{init_time_type_dict[args.init_time_type][0]:02d}/{args.init_time}.{file_suffix}"
    file_hist_pl = f"{args.raw_data_root_path}/pl/{init_time_type_dict[args.init_time_type][0]:02d}/{args.init_time}.{file_suffix}"
    file_init_sfc = f"{args.raw_data_root_path}/sfc/{init_time_type_dict[args.init_time_type][1]:02d}/{args.init_time}.{file_suffix}"
    file_init_pl = f"{args.raw_data_root_path}/pl/{init_time_type_dict[args.init_time_type][1]:02d}/{args.init_time}.{file_suffix}"
    print(file_hist_sfc)
    data, _ = make_hers_input_merge(file_hist_sfc, file_hist_pl, file_init_sfc, file_init_pl, args.save_dir,
                                    args.tpsetzero, args.raw_data_format_type)

    print("data.level.values:", data.level.values)
    print("data.time.values:", data.time.values)
    print("data:", data)
    print_level_info(data)
    timer.print_elapsed_time("processing time")
