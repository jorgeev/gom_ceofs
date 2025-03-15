# %%
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean as cmo
import time

ds = xr.open_dataset('/LUSTRE/ID/hycom_jvz/CEOFS_anomalies_7d.nc')
eigvals = np.load('/LUSTRE/ID/hycom_jvz/eigvals_wstd_k20_reg_no_caribbean.npy')
eigvecs = np.load('/LUSTRE/ID/hycom_jvz/eigvecs_wstd_k20_reg_no_caribbean.npy')
C = np.load('/LUSTRE/ID/hycom_jvz/H_hilbert_wstd_reg_no_caribbean.npy')
plon = np.load('/LUSTRE/ID/hycom_jvz/coords_plon.npy')
plat = np.load('/LUSTRE/ID/hycom_jvz/coords_plat.npy')
mask = ds['mask'].data
mask[:120, 250:] = np.nan
mask[:, 480:] = np.nan
mask = np.where(np.isnan(mask), 0, mask)
upper_layer_shape = ds['top_layer'].shape

# Sort the eigenvalues and eigenvectors 
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

print(f"eigvals shape: {eigvals.shape}")
print(f"eigvecs shape: {eigvecs.shape}")
print(f"C shape: {C.shape}")
print(f"mask shape: {mask.shape}")
print(f"upper_layer shape: {upper_layer_shape}")

# %% Explained variance
Totvar = np.sum(eigvals)
percent_var = 100*(eigvals/Totvar)

# %% Plot the eigenvalues variance

plot_expvar = False
expvar_df = pd.DataFrame(percent_var, index=np.arange(1, eigvals.shape[0]+1), columns=['expvar'])

if plot_expvar:
    # Plot the eigenvalues
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, eigvals.shape[0]+1), percent_var, 'k')
    plt.xlabel('Index')
    plt.ylabel('% Variance explained')
    plt.grid()
    plt.tight_layout()
    plt.title('Eigenvalues')
    plt.savefig('expvar_full_wstd_reg.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Plot variance uncertainty
    error = eigvals * np.sqrt(2/eigvals.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, eigvals.shape[0]+1), eigvals, 'k')
    plt.errorbar(np.arange(1, eigvals.shape[0]+1), eigvals, yerr=error, fmt='o', color='k')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalues uncertainty')
    plt.grid()
    plt.tight_layout()
    plt.title('Eigenvalues uncertainty')
    plt.savefig('expvar_full_wstd_reg_error.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # Compute the cumulative sum of the eigenvalues
    expvar_cumsum = np.cumsum(percent_var)
    expvar_df["cumsum"] = expvar_cumsum

    # Plot the cumulative sum of the eigenvalues
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, eigvals.shape[0]+1), expvar_cumsum, 'k')
    plt.xlabel('Index')
    plt.ylabel('Cumulative sum of variance explained')
    plt.grid()
    plt.tight_layout()
    plt.title('Cumulative sum of eigenvalues')
    plt.savefig('expvar_cumsum_full_wstd_reg.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    expvar_df.to_csv('expvar_full_wstd_reg.csv', index=True)

# %% Spatial modes
eigvals = eigvals[:20]
eigvecs = eigvecs[:, :20]
aux_vec = C @ eigvecs
EOFs = aux_vec / np.linalg.norm(aux_vec)

CP = C.conj().T @ EOFs

print(f"CP shape: {CP.shape}")
print(f"EOFs (spatial_eigvecs) shape: {EOFs.shape}")

# %% Unitary eigenvectors
unit_EOFs = np.zeros_like(EOFs)
unit_CP = np.zeros_like(CP)
for ii in range(EOFs.shape[1]):
    unit_EOFs[:, ii] = EOFs[:, ii] * np.sqrt(eigvals[ii])
    unit_CP[:, ii] = CP[:, ii] / np.sqrt(eigvals[ii])

# %% Allocate the matrix for the reconstruction
EOF_upper_layer = np.zeros((EOFs.shape[1], mask.shape[0], mask.shape[1]), dtype=np.complex128)
EOF_lower_layer = np.zeros((EOFs.shape[1], mask.shape[0], mask.shape[1]), dtype=np.complex128)
print(f"EOF_upper_layer shape: {EOF_upper_layer.shape}")

# Fill the matrix with the reconstruction and compute the amplitude and phase maps
print('Filling the matrix with the reconstruction...')
valid_pixels = np.sum(mask).astype(int)
aux_mask = mask.copy().astype(bool)
for ii in range(EOFs.shape[1]):
    EOF_upper_layer[ii, aux_mask] = unit_EOFs[:valid_pixels, ii]
    EOF_lower_layer[ii, aux_mask] = unit_EOFs[valid_pixels:, ii]

print('Computing the amplitude and phase maps...')
map_upper_layer_amp = np.abs(EOF_upper_layer)
map_upper_layer_phase = np.angle(EOF_upper_layer)
map_lower_layer_amp = np.abs(EOF_lower_layer)
map_lower_layer_phase = np.angle(EOF_lower_layer)
EOF_upper_layer = EOF_upper_layer.real
EOF_lower_layer = EOF_lower_layer.real

# %% EOF maps

plot_EOFs = False
if plot_EOFs:
    print('Plotting the EOF maps...')
    for mode in range(20):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
        ax[0].set_extent([-98, -82, 18.5, 30], crs=ccrs.PlateCarree())
        aux_mask = mask.astype(float)
        aux_mask[aux_mask == 0] = np.nan
        f1 = ax[0].pcolormesh(plon, plat, EOF_upper_layer[mode, :, :]*aux_mask, cmap=cmo.cm.balance)
        ax[0].coastlines()
        ax[0].set_title('Upper layer')
        # Horizontal colorbar
        plt.colorbar(f1, orientation='horizontal', pad=0.05, aspect=50)
        f2 = ax[1].pcolormesh(plon, plat, EOF_lower_layer[mode, :, :]*aux_mask, cmap=cmo.cm.balance)
        ax[1].coastlines()
        ax[1].set_title('Lower layer')
        plt.colorbar(f2, orientation='horizontal', pad=0.05, aspect=50)
        fig.tight_layout()
        fig.suptitle(f'EOF Mode {mode+1}')
        plt.savefig(f'NoCarib_full/EOF_mode_{mode+1}_rescaled.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        ax[0].clear()
        ax[1].clear()
        plt.close()
    fig.clf()

# %% Amplitude and phase maps quiver plot
plot_quivermaps = False
if plot_quivermaps:
    u_upper =  np.sin(map_upper_layer_phase) * mask * map_upper_layer_amp
    v_upper =  np.cos(map_upper_layer_phase) * mask * map_upper_layer_amp
    u_lower =  np.sin(map_lower_layer_phase) * mask * map_lower_layer_amp
    v_lower =  np.cos(map_lower_layer_phase) * mask * map_lower_layer_amp

    print('Plotting the amplitude and phase maps...')
    for mode in range(20):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
        ax[0].set_extent([-98, -82, 18.5, 30], crs=ccrs.PlateCarree())
        aux_mask = mask.astype(float)
        aux_mask[aux_mask == 0] = np.nan
        # Compare maxplitude of upper and lower layer
        max_amp_upper = np.max(map_upper_layer_amp[mode])
        max_amp_lower = np.max(map_lower_layer_amp[mode])
        # Keep the maximum amplitude
        max_amp = np.max([max_amp_upper, max_amp_lower])

        f1 = ax[0].pcolormesh(plon, plat, map_upper_layer_amp[mode]/max_amp*aux_mask, cmap=cmo.cm.tempo, vmin=0, vmax=1)
        ax[0].quiver(plon[::8, ::8], plat[::8, ::8], u_upper[mode, ::8, ::8], v_upper[mode, ::8, ::8], color='k', scale=200, pivot='middle')
        ax[0].set_title('Upper layer amplitude and phase')
        plt.colorbar(f1, orientation='horizontal', pad=0.05, aspect=50)
        f2 = ax[1].pcolormesh(plon, plat, map_lower_layer_amp[mode]/max_amp*aux_mask, cmap=cmo.cm.tempo, vmin=0, vmax=1)
        ax[1].quiver(plon[::8, ::8], plat[::8, ::8], u_lower[mode, ::8, ::8], v_lower[mode, ::8, ::8], color='k', scale=200, pivot='middle')
        ax[1].set_title('Lower layer amplitude and phase')
        plt.colorbar(f2, orientation='horizontal', pad=0.05, aspect=50)
        fig.tight_layout()
        fig.suptitle(f'Mode {mode+1} Amplitude and Phase, max_amp = {max_amp:.2f}')
        plt.savefig(f'hase_amp_mode_{mode+1}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    del u_upper, v_upper, u_lower, v_lower
    fig.clf()


# %% Amplitue and phase maps
plot_amp_phase = False
if plot_amp_phase:
    print('Plotting the amplitude and phase maps...')
    for mode in range(20):
        fig, ax = plt.subplots(2, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()}, sharex=True, sharey=True)
        ax[0].set_extent([-98, -82, 18.5, 30], crs=ccrs.PlateCarree())
        aux_mask = mask.astype(float)
        aux_mask[aux_mask == 0] = np.nan
        f1 = ax[0, 0].pcolormesh(map_upper_layer_amp[mode, :, :]*aux_mask, cmap=cmo.cm.deep)
        ax[0, 0].set_title('Upper layer amplitude')
        plt.colorbar(f1)
        f2 = ax[0, 1].pcolormesh(map_upper_layer_phase[mode, :, :]*aux_mask, cmap="gray")
        ax[0, 1].set_title('Upper layer phase')
        plt.colorbar(f2)
        f3 = ax[1, 0].pcolormesh(map_lower_layer_amp[mode, :, :]*aux_mask, cmap=cmo.cm.deep)
        ax[1, 0].set_title('Lower layer amplitude')
        plt.colorbar(f3)
        f4 = ax[1, 1].pcolormesh(map_lower_layer_phase[mode, :, :]*aux_mask, cmap="gray")
        ax[1, 1].set_title('Lower layer phase')
        plt.colorbar(f4)
        fig.tight_layout()
        fig.suptitle(f'Mode {mode+1} Amplitude and Phase')
        plt.savefig(f'NoCarib_full/phase_amp_mode_{mode+1}.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    fig.clf()

# %% Trying to reconstruct the first EOF

data = unit_EOFs[:, :5] @ unit_CP[1000, :5]
rec_upper = np.empty_like(mask)
rec_lower = np.empty_like(mask)
rec_upper[mask==1] = data[:valid_pixels]
rec_lower[mask==1] = data[valid_pixels:]

fig, ax = plt.subplots(2, 2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
ax[0, 0].set_extent([-98, -82, 18.5, 30], crs=ccrs.PlateCarree())
ax[0, 0].pcolormesh(plon, plat, rec_upper*aux_mask, cmap=cmo.cm.balance)
ax[0, 0].coastlines()
ax[0, 0].set_title('Reconstructed upper layer')
ax[0, 1].pcolormesh(plon, plat, rec_lower*aux_mask, cmap=cmo.cm.balance)
ax[0, 1].coastlines()
ax[0, 1].set_title('Reconstructed lower layer')
ax[1, 0].pcolormesh(plon, plat, ds['top_layer'][3, :, :].data*aux_mask, cmap=cmo.cm.balance)
ax[1, 0].coastlines()
ax[1, 0].set_title('Original upper layer')
ax[1, 1].pcolormesh(plon, plat, ds['bottom_layer'][3, :, :].data*aux_mask, cmap=cmo.cm.balance)
ax[1, 1].coastlines()
ax[1, 1].set_title('Original lower layer')
fig.tight_layout()
fig.show()
# %%

plt.pcolormesh(plon, plat, ds['bottom_layer'][0, :, :].data*aux_mask, cmap=cmo.cm.balance)
# %%

