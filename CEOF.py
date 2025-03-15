# %% Libraries
import xarray as xr
import numpy as np
import scipy as sp
import time
import dask.array as da

import matplotlib.pyplot as plt
import cmocean as cmo

# %% Load the data
print('Loading the data...')
start_time = time.time()
ds = xr.open_dataset('/LUSTRE/ID/hycom_jvz/CEOFS_anomalies_7d.nc').load()
list_vars = ['top_layer', 'bottom_layer', 'ssh']
print(f'Data loaded in {time.time() - start_time} seconds')

# Refine mask to remove caribbean
caribbean = False
if not caribbean:
    print('Removing caribbean sea...')
    mask = ds['mask'].data
    mask[:120, 250:] = np.nan
    mask[:, 480:] = np.nan
    mask = np.where(np.isnan(mask), 0, mask)

# Refine the mean aftter 
check_mean = True
if check_mean:
    print('Ensuring mean is zero...')

    # Ensure mean is zero
    daily_clim = ds.groupby('time.dayofyear').mean(dim='time')
    anomalies = ds.copy()

    # Compute anomalies by subtracting daily climatology
    anomalies["bottom_layer"] = ds["bottom_layer"].groupby('time.dayofyear') - daily_clim["bottom_layer"]
    anomalies["top_layer"] = ds["top_layer"].groupby('time.dayofyear') - daily_clim["top_layer"]
    anomalies["mask"] = ds["mask"]
    ds = anomalies
    print(ds)

# %% Extract the valid pixels and mask
print('Extracting the valid pixels and mask...')

valid_pixels = np.nansum(mask).astype(int)
mask = mask.astype(bool)
print('Valid pixels and mask extracted')

# %% Allocat the matrix for Hilbert EOFs
print('Allocating the matrix for Hilbert EOFs...')
nvars = len(list_vars)
H = np.zeros((ds.time.shape[0], valid_pixels * nvars))
print('Matrix allocated')

# Fill the matrix with bottom and top layer anomalies
print('Filling the matrix with bottom and top layer anomalies...')
time_start = time.time()
for ii in range(ds.time.shape[0]):
    if nvars == 2:
        H[ii, :valid_pixels] = ds[list_vars[0]].isel(time=ii).data[mask]
        H[ii, valid_pixels:] = ds[list_vars[1]].isel(time=ii).data[mask]
    elif nvars == 3:
        H[ii, :valid_pixels] = ds[list_vars[0]].isel(time=ii).data[mask]
        H[ii, valid_pixels:2*valid_pixels] = ds[list_vars[1]].isel(time=ii).data[mask]
        H[ii, 2*valid_pixels:] = ds[list_vars[2]].isel(time=ii).data[mask]
print(f'Matrix filled in {time.time() - time_start} seconds')

# Remove the first and last 3 time steps since empty due to the rolling mean
print('Removing the first and last 3 time steps...')
H = H[3:-3, :]
H = H.T
print('First and last 3 time steps removed')

# Save original matrix
print('Saving the original matrix...')
np.save(f'/LUSTRE/ID/hycom_jvz/H_wstd_reg_no_caribbean_nvars{nvars}.npy', H)
print('Original matrix saved')

# Verify if there ins nan values
if np.isnan(H).sum() != 0:
    print('There are nan values in the matrix')
    exit()
else:
    print('There are no nan values in the matrix')

#%% Hilbert Transform
print('Computing the Hilbert Transform...')
start_time = time.time()
H_hilbert = sp.signal.hilbert(H, axis=1)
end_time = time.time()
print(f"Hilbert matrix shape: {H_hilbert.shape}")
print(f'Hilbert Transform computed in {end_time - start_time} seconds')

# Save the Hilbert matrix
print('Saving the Hilbert matrix...')
np.save(f'/LUSTRE/ID/hycom_jvz/H_hilbert_wstd_reg_no_caribbean_nvars{nvars}.npy', H_hilbert)
print('Hilbert matrix saved')

# %% Compute covariance matrix
print('Computing the covariance matrix...')
#H_dask = da.from_array(H_hilbert, chunks=(1000, 10000))
start_time = time.time()
C = H_hilbert.conj().T @ H_hilbert # Time covariance
C /= (H_hilbert.shape[0] - 1) # Normalize by the number of spatial points-1
# Ridge regularization
C += 1e-6 * np.eye(C.shape[0])
Cn = np.linalg.cond(C)
print(f'Condition number: {Cn}')
#C = C.compute().values
end_time = time.time()
print(f'Covariance matrix shape: {C.shape}')
print(f'Covariance matrix computed in {end_time - start_time} seconds')

# Save the covariance matrix
print('Saving the covariance matrix...')
np.save(f'/LUSTRE/ID/hycom_jvz/C_wstd_reg_no_caribbean_nvars{nvars}.npy', C)
print('Covariance matrix saved')

# %% Decompose the Hilbert matrix
print('Decomposing the Hilbert matrix...')
C = np.load(f'/LUSTRE/ID/hycom_jvz/C_wstd_reg_no_caribbean_nvars{nvars}.npy')
# Compute the temporal modes
start_time = time.time()
k = 20
eigvals, eigvecs = sp.sparse.linalg.eigsh(C, k=k, which='LM')
eigvals_F, eigvecs_F = sp.linalg.eigh(C)
end_time = time.time()
print(f'Singular values computed in {end_time - start_time} seconds')

# Save the singular values
print('Saving the singular values...')
np.save(f'/LUSTRE/ID/hycom_jvz/eigvals_wstd_k{k}_reg_no_caribbean_nvars{nvars}.npy', eigvals)
np.save(f'/LUSTRE/ID/hycom_jvz/eigvecs_wstd_k{k}_reg_no_caribbean_nvars{nvars}.npy', eigvecs)
print('Singular values saved')

# Save the full eigenvalues
print('Saving the full eigenvalues...')
np.save(f'/LUSTRE/ID/hycom_jvz/eigvals_full_wstd_reg_no_caribbean_nvars{nvars}.npy', eigvals_F)
np.save(f'/LUSTRE/ID/hycom_jvz/eigvecs_full_wstd_reg_no_caribbean_nvars{nvars}.npy', eigvecs_F)
print('Full eigenvalues saved')

# %%
