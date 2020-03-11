#=========================================================================
# This is a simple function to define land and ocean masks for LMR output.
#   author: Michael P. Erb
#   date  : October 19, 2016
#=========================================================================

import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans


# Define land and ocean masks for LMR output:
def masks(file_name):
    #
    print("This masking function doesn't work correctly near the poles and international date line. Double-check the mask.")
    #
    handle = np.load(file_name)
    lon = handle['lon']
    lat = handle['lat']
    handle.close()
    #
    # Make a version of lon where western hemisphere longitudes are negative
    lon_we = lon
    lon_we[lon_we>180] = lon_we[lon_we>180]-360
    #
    # Make an ocean mask
    allpoints = np.ones((lat.shape[0],lon_we.shape[1]))
    oceanmask = maskoceans(lon_we,lat,allpoints,inlands=False).filled(np.nan)
    oceanmask[0,:] = 1
    #
    # Make a land mask
    landmask = np.zeros((lat.shape[0],lon_we.shape[1]))
    landmask[:] = np.nan
    landmask[np.isnan(oceanmask)] = 1
    #
    return landmask, oceanmask


# Define land and ocean masks for netcdf files:
def masks_netcdf(file_name):
    handle = xr.open_dataset(file_name,decode_times=False)
    lon_1d = handle['lon'].values
    lat_1d = handle['lat'].values
    #
    # Make 2d versions of lat and lon
    lon,lat = np.meshgrid(lon_1d,lat_1d)
    #
    # Make a version of lat where western hemisphere latitudes are negative
    lon_we = lon
    lon_we[lon_we>180] = lon_we[lon_we>180]-360
    #
    # Make an ocean mask
    allpoints = np.ones((lat.shape[0],lon_we.shape[1]))
    oceanmask = maskoceans(lon_we,lat,allpoints,inlands=False).filled(np.nan)
    oceanmask[0,:] = 1
    #
    # Make a land mask
    landmask = np.zeros((lat.shape[0],lon_we.shape[1]))
    landmask[:] = np.nan
    landmask[np.isnan(oceanmask)] = 1
    #
    return landmask, oceanmask

