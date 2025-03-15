## Export general utils and path configurations
from hycom.io import *
from hycom.info import *
from hycom.tools import *

import matplotlib.pyplot as plt
import cmocean.cm as cmo

from os.path import join
import numpy as np
import scipy as cp
import dask
import xarray as xr
import pandas as pd
from glob import glob

from multiprocessing import Pool
from scipy.signal import detrend
import os

coast_line =  '/LUSTRE/CIGOM/HYCOM/GOMl0.04/topo/Linea_Costa.dat'
rt = '/LUSTRE/CIGOM/HYCOM/GOM_V4/GOMl0.04/expt_01.0/data'

keep_std = False

idm = 385
jdm = 541
kdm = 36
year0 = 1993
year1 = 2019
yy0 = 93
yy1 = 119
years = np.arange(year0, year1+1)
yys = np.arange(yy0, yy1+1)
topl = [ii for ii in range(0, 24)]
botl = [ii for ii in range(32, 35)]
files_dict = {}
all_files = []
## Concatenate all the paths
for ii, yy in enumerate(yys):
    data_files = sorted(glob(join(rt, f'tarv_{yy:03d}', f'010_archv.{years[ii]}_*.a')))
    files_dict[f'{yy:03d}'] = data_files
    all_files.extend(data_files)
coast_line = pd.read_csv(coast_line, sep='\s+', header=None)

# Read coordinates
coords = read_hycom_coords(join(rt, 'regional.grid.a'), fields=['plon:', 'plat:', 'qlon:', 'qlat:', 'ulon:', 'ulat:', 'vlon:', 'vlat:'])
file_depth = join(rt, 'regional.depth.a')

def process_file(args):
    """
    Process a single file and return the top and bottom layer sums
    """
    file_index, file = args
    tkns = read_hycom_fields(join(rt, file), ['thknss'])['thknss'] / 9806
    ssh = read_hycom_fields(join(rt, file), ['srfhgt'])['srfhgt']
    top = np.nansum(tkns[:24, :, :], axis=0)
    bottom = np.nansum(tkns[32:34, :, :], axis=0)
    return file_index, top, bottom, ssh

def read_depth_hycom(im, jm, file, demo=False):
    """
    Reads bathymetry data from a binary file and processes it.

    Parameters:
    im (int): Number of columns (IDM).
    jm (int): Number of rows (JDM).
    file (str): Path to the bathymetry file.

    Returns:
    numpy.ndarray: 2D array of bathymetry data with masked values as NaN.
    """
    IJDM = im * jm

    # Open the file and read the binary data
    with open(file, 'rb') as depth_fid:
        bathy = np.fromfile(depth_fid, dtype='>f4', count=IJDM)  # '>f4' for big-endian float32

    # Mask bathymetry values greater than 1e20 and reshape
    bathy[bathy > 1e20] = np.nan
    bathy = bathy.reshape((im, jm))  # Reshape to (JDM, IDM)

    if demo:
        import matplotlib.pyplot as plt
        import cmocean.cm as cmo
        plt.imshow(bathy, cmap=cmo.deep)
        plt.colorbar()
        plt.show()

    return bathy

bathy = read_depth_hycom(idm, jdm, file_depth, demo=False)

mask1 = bathy/bathy
mask2 = ~np.isnan(bathy)

# Get index for values where depth < 1000
idx = np.where(bathy < 1000)
# Mask the values
depth_1000 = bathy.copy()
depth_1000[idx] = np.nan
depth_1000 = depth_1000 / depth_1000

# Initialize the xarray dataset

start_date = pd.Timestamp(f'{year0}-01-01')
end_date = pd.Timestamp(f'{year1}-12-31')

dates=pd.date_range(start_date, end_date, freq='D')
# Array should be 3D (time, lat, lon) or date, idm, jdm

top_layer = np.empty((dates.size, idm, jdm))
bottom_layer = np.empty((dates.size, idm, jdm))
ssh = np.empty((dates.size, idm, jdm))

ds = xr.Dataset(data_vars=dict(top_layer=(('time', 'lat', 'lon'), top_layer), 
                               bottom_layer=(('time', 'lat', 'lon'), bottom_layer),
                               ssh=(('time', 'lat', 'lon'), ssh)),
                coords=dict(
                    time=(('time'),dates), lat=(('lat'),np.arange(idm)), lon=(('lon'),np.arange(jdm))
                    ))

# Number of processes - use number of CPU cores minus 1 to leave one core free
num_processes = max(os.cpu_count() - 2, 1)

# Create list of arguments for each file
file_args = [(i, file) for i, file in enumerate(all_files)]

# Create process pool and map the work
with Pool(processes=num_processes) as pool:
    results = pool.map(process_file, file_args)

# Sort results by index and assign to dataset
for file_index, top, bottom, ssh in sorted(results):
    ds['top_layer'][file_index, :, :] = top
    ds['bottom_layer'][file_index, :, :] = bottom
    ds['ssh'][file_index, :, :] = ssh.reshape(idm, jdm)
# Remove nans
ds['ssh'].data = np.where(np.isnan(ds['ssh'].data), 0, ds['ssh'].data)

# Compute daily climatology
daily_clim = ds.groupby('time.dayofyear').mean(dim='time')
if keep_std:
    daily_std = ds.groupby('time.dayofyear').std(dim='time')

# Compute anomalies by subtracting daily climatology
anomalies = (ds.groupby('time.dayofyear') - daily_clim).groupby('time.dayofyear')
if keep_std:
    anomalies = anomalies / daily_std
    del daily_std
del daily_clim, ds

# Apply detrend to anomalies
anomalies['ssh'].data = np.where(np.isnan(anomalies['ssh'].data), 0, anomalies['ssh'].data)
anomalies['top_layer'].data = np.where(np.isnan(anomalies['top_layer'].data), 0, anomalies['top_layer'].data)
anomalies['bottom_layer'].data = np.where(np.isnan(anomalies['bottom_layer'].data), 0, anomalies['bottom_layer'].data)

top_detrend = detrend(anomalies['top_layer'], axis=0)
bottom_detrend = detrend(anomalies['bottom_layer'], axis=0)
ssh_detrend = detrend(anomalies['ssh'], axis=0)

detrended = anomalies.copy()
detrended['top_layer'].data = top_detrend
detrended['bottom_layer'].data = bottom_detrend
detrended['ssh'].data = ssh_detrend

# Keep only deep water pixels
detrended['top_layer'] *= depth_1000
detrended['bottom_layer'] *= depth_1000
detrended['ssh'] *= depth_1000

# Compute 7 day running mean
anomalies_7d = detrended.rolling(time=7, center=True).mean(dim='time')
anomalies_7d['mask'] = (('lat', 'lon'), depth_1000)

if keep_std:
    anomalies_7d.to_netcdf('/LUSTRE/ID/hycom_jvz/CEOFS_anomalies_7d_with_std.nc')
else:
    anomalies_7d.to_netcdf('/LUSTRE/ID/hycom_jvz/CEOFS_anomalies_7d.nc')

if keep_std:
    anomalies_7d['top_layer'].isel(time=1000).plot.pcolormesh(cmap=cmo.balance)
    plt.savefig('Test_CEOFS_anomalies_7d_with_std.png')
else:
    anomalies_7d['top_layer'].isel(time=1000).plot.pcolormesh(cmap=cmo.balance)
    plt.savefig('Test_CEOFS_anomalies_7d.png')