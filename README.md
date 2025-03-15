# gom_ceofs
Hilbert EOF for the Gulf of Mexico

In this repsoitory I store the codes used to compute the coupled variability of of the different variables from HYCOM

The may rutine is splited across several codes

1. [preprocess.py](preporcess.py). Reads the hycom `*.ab` outputs and create a netcdf with the anomalies
  - This code subtract the mean, detreand and pass a 7 day rolling mean to the full time series (in that order).
  - The code processes layer thikness (`thknss`) and the sea surface height (`srfhgt`) variables but can be expanded to more varibles.
2. [CEOF.py](CEOF.py). Computes the Hilbert CEOF coupled modes for a given number of variables, rightnow it performes the extraction of the 20 larges modes and the full decomposition.
3. [recosntruction.py](reconstruction.py). Compute the unitary PCA and EOF for the eigenvalues and eigenvector, also the level of explained variance.
