#==================================================================================
# This script computes mean timeseries over regions of the US.  In particular, is
# computes values over the four drought regions examined in Cook et al. 2014.
#   author: Michael P. Erb
#   date  : January 10, 2017
#==================================================================================

import numpy as np
#import mpe_functions as mpe

def compute_US_means(variable,lat,lon):
    #
    # Initialize dictionaries
    variable_means = {}
    #
    # Compute the mean over the entire region and a variety of other regions, including those analysed in Cook et al. 2014
    variable_means['entire']    = compute_means(variable,lat,lon,min(lat),max(lat),min(lon),max(lon))
    variable_means['US']        = compute_means(variable,lat,lon,30,49,-130, -65)  # 30-49N, 130- 65W
    variable_means['southwest'] = compute_means(variable,lat,lon,32,40,-125,-105)  # 32-40N, 125-105W
    variable_means['central']   = compute_means(variable,lat,lon,34,46,-102, -92)  # 34-46N, 102- 92W
    variable_means['northwest'] = compute_means(variable,lat,lon,42,50,-125,-110)  # 42-50N, 125-110W
    variable_means['southeast'] = compute_means(variable,lat,lon,30,39, -92, -75)  # 30-39N,  92- 75W
    #
    return variable_means


def compute_means(variable,lat,lon,lat_min,lat_max,lon_min,lon_max):
    #
    # Replace all 0 values with nan
    variable[variable==0] = np.nan
    #
    # If the longitude axis is positive, make all longitude values positive
    if min(lon) >= 0:
        if lon_min < 0:
           lon_min = lon_min+360
           lon_max = lon_max+360
    #
    lon_2d,lat_2d = np.meshgrid(lon,lat)
    j_indices = np.where((lat>=lat_min) & (lat<=lat_max))[0]
    i_indices = np.where((lon>=lon_min) & (lon<=lon_max))[0]
    nlat = len(j_indices)
    nlon = len(i_indices)
    ntime = variable.shape[0]
    #
    # Select the grid cells in the region
    lat_2d_selected = lat_2d[j_indices[:,None],i_indices[None,:]]
    lon_2d_selected = lon_2d[j_indices[:,None],i_indices[None,:]]
    variable_selected = np.squeeze(variable[:,j_indices[None,:,None],i_indices[None,None,:]])
    #plt.contourf(lon_2d_selected,lat_2d_selected,variable_selected[0,:,:])
    #
    # Put lat and lon on the same dimension
    lat_2d_selected_flatten = np.reshape(lat_2d_selected,(nlat*nlon))
    variable_selected_flatten = np.reshape(variable_selected,(ntime,nlat*nlon))
    lat_selected_weights = np.cos(np.radians(lat_2d_selected_flatten))
    #
    # Compute means, year by year
    variable_mean = np.zeros(variable.shape[0]); variable_mean[:] = np.nan
    for i in range(ntime):
        variable_for_time = variable_selected_flatten[i,:]
        lat_weights_for_time = lat_selected_weights[:]
        variable_mean[i] = np.ma.average(variable_for_time[~np.isnan(variable_for_time)],axis=0,weights=lat_weights_for_time[~np.isnan(variable_for_time)])
    #
    return variable_mean

