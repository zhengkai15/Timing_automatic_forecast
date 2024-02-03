import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# __all__ = ["save_like"]

pl_names = ['z', 't', 'u', 'v', 'r']
sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def weighted_rmse(out, tgt):
    wlat = np.cos(np.deg2rad(tgt.lat))
    wlat /= wlat.mean()
    error = ((out - tgt) ** 2 * wlat)
    return np.sqrt(error.mean(('lat', 'lon')))


def split_variable(ds, name):
    if name in sfc_names:
        v = ds.sel(level=[name])
        v = v.assign_coords(level=[0])
        v = v.rename({"level": "level0"})
        v = v.transpose('member', 'level0', 'time', 'dtime', 'lat', 'lon')
    elif name in pl_names:
        level = [f'{name}{l}' for l in levels]
        v = ds.sel(level=level)
        v = v.assign_coords(level=levels)
        v = v.transpose('member', 'level', 'time', 'dtime', 'lat', 'lon')
    return v

def save_like_od_version(output, input, step, save_dir="", freq=6, split=False):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.chmod(save_dir, 0o777)
        step = (step+1) * freq
        init_time = pd.to_datetime(input.time.values[-1])

        ds = xr.DataArray(
            output[None],
            dims=['time', 'step', 'level', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                step=[step],
                level=input.level,
                lat=input.lat.values,
                lon=input.lon.values,
            )
        ).astype(np.float32)

        if split:
            def rename(name):
                if name == "tp":
                    return "TP06"
                elif name == "r":
                    return "RH"
                return name.upper()

            new_ds = []
            for k in pl_names + sfc_names:
                v = split_variable(ds, k)
                v.name = rename(k)
                new_ds.append(v)
            ds = xr.merge(new_ds, compat="no_conflicts")

        save_name = os.path.join(save_dir, f'{step:03d}.nc')
        # print(f'Save to {save_name} ...')
        ds.to_netcdf(save_name)

def visualize(save_name, vars=[], titles=[], vmin=None, vmax=None):
    import cartopy.crs as ccrs
    fig, ax = plt.subplots(len(vars), 1, figsize=(8, 6), subplot_kw={
                           "projection": ccrs.PlateCarree()})

    def plot(ax, v, title):
        v.plot(
            ax=ax,
            x='lon',
            y='lat',
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            add_colorbar=False
        )
        # ax.coastlines()
        ax.set_title(title)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

    for i, v in enumerate(vars):
        if len(vars) == 1:
            plot(ax, v, titles[i])
        else:
            plot(ax[i], v, titles[i])

    plt.savefig(save_name, bbox_inches='tight',
                pad_inches=0.1, transparent='true', dpi=200)
    plt.close()


def test_visualize(step, data_dir):
    src_name = os.path.join(data_dir, f"{step:03d}.nc")
    ds = xr.open_dataarray(src_name).isel(time=0)
    ds = ds.sel(lon=slice(90, 150), lat=slice(50, 0))
    print(ds)
    u850 = ds.sel(level='U850', step=step)
    v850 = ds.sel(level='V850', step=step)
    ws850 = np.sqrt(u850 ** 2 + v850 ** 2)
    visualize(f'ws850/{step:03d}.jpg', [ws850], [f'20230725-18+{step:03d}h'], vmin=0, vmax=30)


def test_rmse(output_name, target_name):
    output = xr.open_dataarray(output_name)
    output = output.isel(time=0).sel(step=120)
    target = xr.open_dataarray(target_name)

    for level in ["z500", "t850", "t2m", "u10", "v10", "msl", "tp"]:
        out = output.sel(level=level)
        tgt = target.sel(level=level)
        rmse = weighted_rmse(out, tgt).load()
        print(f"{level.upper()} 120h rmse: {rmse:.3f}")




def plot_z500(src_name):
    da = xr.open_dataarray(src_name)
    lat = da.coords["lat"].values
    lon = da.coords["lon"].values
    times = da.coords["time"].values
    LON,LAT = np.meshgrid(lon, lat)
    data1 = da.sel(level="Z500", time=times[0], lat=lat, lon=lon).squeeze().values/9.8
    print(data1.shape)
    print(data1.min(),data1.max())
    levels = np.arange(4880,6048,40)
    fig, ax = plt.subplots(layout='constrained', figsize=(10,5), dpi=300)
    cs = ax.contour(LON, LAT, data1, levels=levels, linewidths=1, cmap="coolwarm", origin="upper")
    ax.clabel(cs,inline=True,fontsize=10)
    ax.set(xlabel="longitude", ylabel="latitude")
    ax.set_title("072_500hpa")
    fig.savefig("072_500hpa.png", dpi=300)

__all__ = ["make_sample", "print_dataarray"]

pl_names = [
    ('geopotential', 'z'),
    ('temperature', 't'),
    ('u_component_of_wind', 'u'),
    ('v_component_of_wind', 'v'),
    ('specific_humidity', 'q'),
]

sfc_names = [
    ('2m_temperature', 't2m'),
    ('2m_dewpoint_temperature', 'd2m'),
    ('sea_surface_temperature', 'sst'),
    ('10m_u_component_of_wind', 'u10m'),
    ('10m_v_component_of_wind', 'v10m'),
    ('100m_u_component_of_wind', 'u100m'),
    ('100m_v_component_of_wind', 'v100m'),
    ('mean_sea_level_pressure', 'msl'),
    ('surface_pressure', 'sp'),
]

avg_names = [
    ('surface_net_solar_radiation', 'ssr'),
    ('surface_solar_radiation_downwards', 'ssrd'),
    ('total_sky_direct_solar_radiation_at_surface', 'fdir'),
    ('top_net_thermal_radiation', 'ttr'),
    ('total_precipitation', 'tp'),
]


def is_pressure(short_name):
    return short_name in ["z", "t", "u", "v", "q"]


def get_channel(pl_names):
    channel = []
    for (_, short_name) in pl_names:
        channel += [f'{short_name}{l}' for l in levels]
    for (_, short_name) in sfc_names + avg_names:
        channel += [short_name]
    return channel


def make_sample(data_dir, version="c79"):
    new_pl_names = pl_names
    if version == "c92":
        new_pl_names += [('specific_cloud_liquid_water_content', 'clwc')]

    ds = []
    for (long_name, short_name) in new_pl_names + sfc_names:
        file_name = os.path.join(data_dir, f"{long_name}.nc")
        v = xr.open_dataarray(file_name)
        if is_pressure(short_name) and v.level.values[0] != 50:
            v = v.reindex(level=v.level[::-1])

        if short_name in ["q", "clwc"]:
            print(f"Convert {short_name} to g/kg")
            v = v * 1000

        v.name = "data"
        v.attrs = {}
        ds.append(v)

    for (long_name, short_name) in avg_names:
        zero = v * 0
        print(zero)
        print(f"zero: {zero.min():.3f} ~ {zero.max():.3f}")
        ds.append(zero)

    ds = xr.concat(ds, 'level').rename({"level": "channel"})
    ds = ds.assign_coords(channel=get_channel(new_pl_names))
    return ds


def print_dataarray(
        ds, msg='',
        names=["z500", "t850", "q700", "t2m", "d2m", "sst", "msl", "tp"]
):
    v = ds.isel(time=0)
    msg += f"shape: {v.shape}"

    if 'lat' in ds.dims:
        lat = ds.lat.values
        msg += f", lat: {lat[0]:.3f} ~ {lat[-1]:.3f}"
    if 'lon' in ds.dims:
        lon = ds.lon.values
        msg += f", lon: {lon[0]:.3f} ~ {lon[-1]:.3f}"

    if "level" in v.dims and len(v.level) > 1:
        names = np.intersect1d(names, v.level.data)
        for lvl in names:
            x = v.sel(level=lvl).values
            msg += f"\nlevel: {lvl:04d}, value: {x.min():.3f} ~ {x.max():.3f}"

    if "channel" in v.dims and len(v.channel) > 1:
        names = np.intersect1d(names, v.channel.data)
        for ch in names:
            x = v.sel(channel=ch).values
            msg += f"\nchannel: {ch}, value: {x.min():.3f} ~ {x.max():.3f}"

    print(msg)
def save_with_progress(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar

    if 'time' in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    ds = ds.astype(dtype)

    if save_name.endswith("nc"):
        obj = ds.to_netcdf(save_name, compute=False)
    elif save_name.endswith("zarr"):
        obj = ds.to_zarr(save_name, compute=False)

    with ProgressBar():
        obj.compute()

def save_like(output, input, lead_time, save_dir=""):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        init_time = pd.to_datetime(input.time.values[-1])

        ds = xr.DataArray(
            data=output,
            dims=['time', 'lead_time', 'channel', 'lat', 'lon'],
            coords=dict(
                time=[init_time],
                lead_time=[lead_time],
                channel=input.level.values,
                lat=input.lat.values,
                lon=input.lon.values,
            )
        ).astype(np.float32)
        print_dataarray(ds)
        save_name = os.path.join(save_dir, f'{lead_time:03d}.nc')
        save_with_progress(ds, save_name)



