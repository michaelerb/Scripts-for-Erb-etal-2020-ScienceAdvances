####################################################################
#
# The file contains a variety of functions I coded which might be useful.
# An example of how these can be invoked:
#    import mpe_functions as mpe
#    i, j = mpe.lonlat(0,0)
#  - mpe
#
####################################################################

import numpy as np
from mpl_toolkits.basemap import Basemap
import calendar


# Compute the annual-means weighted by the correct number of days in each month.
def annual_mean(years,data_monthly):
    data_annual = np.zeros((len(years)))
    data_annual[:] = np.nan
    for i,year in enumerate(years):
        if calendar.isleap(year):
            days_in_months = [31,29,31,30,31,30,31,31,30,31,30,31]
        else:
            days_in_months = [31,28,31,30,31,30,31,31,30,31,30,31]
        data_annual[i] = np.average(data_monthly[i,:],weights=days_in_months)
    #
    return data_annual

# Compute the tropical year (April-March) annual-means weighted by the correct number of days in each month.
def tropical_year_mean(years,data_monthly):
    data_annual = np.zeros((len(years)))
    data_annual[:] = np.nan
    for i,year in enumerate(years):
        if calendar.isleap(year+1):
            days_in_months = [30,31,30,31,31,30,31,30,31,31,29,31]
        else:
            days_in_months = [30,31,30,31,31,30,31,30,31,31,28,31]
        data_annual[i] = np.average(data_monthly[i,:],weights=days_in_months)
    #
    return data_annual

# Return the indexes closest to the desired lon and lat
def latlon(lat, lon):
    file_data = np.load('/home/scec-00/lmr/erbm/LMR/archive_output/2ka_mlost_ccsm4_pagesall_0.75_fixed_analysis_Ye/r0/ensemble_mean_tas_sfc_Amon.npz')
    file_lat = file_data['lat']
    file_lon = file_data['lon']
    j = np.abs(file_lat[:,0]-lat).argmin()
    i = np.abs(file_lon[0,:]-lon).argmin()
    return j, i

# As above, but in the other order.
def lonlat(lon, lat):
    file_data = np.load('/home/scec-00/lmr/erbm/LMR/archive_output/2ka_mlost_ccsm4_pagesall_0.75_fixed_analysis_Ye/r0/ensemble_mean_tas_sfc_Amon.npz')
    file_lon = file_data['lon']
    file_lat = file_data['lat']
    i = np.abs(file_lon[0,:]-lon).argmin()
    j = np.abs(file_lat[:,0]-lat).argmin()
    return i, j


# Return the x and y values of all of the assimilated proxy records for use in plotting on a Robinson projection
def proxyxy(filename,m):
    assimilated_proxies = np.load(filename)
    lon_proxies = np.zeros(len(assimilated_proxies))
    lat_proxies = np.zeros(len(assimilated_proxies))
    i = 0
    for i in range(len(assimilated_proxies)):
        lon_proxies[i] = assimilated_proxies[i][str(assimilated_proxies[i].keys()).translate(None,'[\']')][2]
        lat_proxies[i] = assimilated_proxies[i][str(assimilated_proxies[i].keys()).translate(None,'[\']')][1]
    #
    x_proxies, y_proxies = m(lon_proxies,lat_proxies)
    return x_proxies, y_proxies


# This function takes a 2d lat-lon variable and computes the global-mean.
def global_mean_2d(variable,lats):
    variable_zonal = np.nanmean(variable,axis=1)
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.average(variable_zonal,axis=0,weights=lat_weights)
    return variable_global


# This is like the last function, but accepts masked files.
def global_mean_2d_masked(variable,lats):
    variable_zonal = np.ma.mean(variable,axis=1)
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.ma.average(variable_zonal,axis=0,weights=lat_weights)
    return variable_global


# This function takes a time-lat-lon variable and computes the global-mean.
# (The first dimension isn't involved in the calculation, so it can be anything.)
def global_mean(variable,lats):
    variable_zonal = np.nanmean(variable,axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.zeros(variable.shape[0])
    variable_global[:] = np.nan
    time = 0
    while time < variable.shape[0]:
        variable_global[time] = np.average(variable_zonal[time,:],axis=0,weights=lat_weights)
        time=time+1
    return variable_global


# This is like the last function, but accepts masked files.
def global_mean_masked(variable,lats):
    variable_zonal = np.ma.mean(variable,axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.zeros(variable.shape[0])
    variable_global[:] = np.nan
    time = 0
    for time in range(variable.shape[0]):
        variable_global[time] = np.ma.average(variable_zonal[time,:],axis=0,weights=lat_weights)
    #
    return variable_global


def spatial_mean_lats(variable,lats,lat_min,lat_max):
    j_min = np.abs(lats-lat_min).argmin()
    j_max = np.abs(lats-lat_max).argmin()
    print('Computing spatial mean. j='+str(j_min)+'-'+str(j_max)+'.  Points are inclusive.')
    variable_zonal = np.nanmean(variable,axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_mean = np.zeros(variable.shape[0])
    variable_mean[:] = np.nan
    time = 0
    for time in range(variable.shape[0]):
        variable_mean[time] = np.average(variable_zonal[time,j_min:j_max+1],axis=0,weights=lat_weights[j_min:j_max+1])
    #
    return variable_mean


# This function takes a time-lat-lon variable and computes the mean for a given range of i and j.
def spatial_mean(variable,lats,j_min,j_max,i_min,i_max):
    print('Computing spatial mean. i='+str(i_min)+'-'+str(i_max)+', j='+str(j_min)+'-'+str(j_max)+'.  Points are inclusive.')
    variable_zonal = np.nanmean(variable[:,:,i_min:i_max+1],axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_mean = np.zeros(variable.shape[0])
    variable_mean[:] = np.nan
    time = 0
    for time in range(variable.shape[0]):
        variable_mean[time] = np.average(variable_zonal[time,j_min:j_max+1],axis=0,weights=lat_weights[j_min:j_max+1])
    #
    return variable_mean



# This function takes a time-lat-lon variable and computes the mean for a given range of lon and lat.
def spatial_mean_latlon(variable,lat,lon,lat_min,lat_max,lon_min,lon_max):
    # Make all longitude values positive
    if lon_min < 0:
        lon_min = lon_min+360
    if lon_max < 0:
        lon_max = lon_max+360
    #
    j_min = np.abs(lat-lat_min).argmin()
    j_max = np.abs(lat-lat_max).argmin()
    i_min = np.abs(lon-lon_min).argmin()
    i_max = np.abs(lon-lon_max).argmin()
    #
    print('Computing spatial mean. i='+str(i_min)+'-'+str(i_max)+', j='+str(j_min)+'-'+str(j_max)+'.  Points are inclusive.')
    variable_zonal = np.nanmean(variable[:,:,i_min:i_max+1],axis=2)
    lat_weights = np.cos(np.radians(lat))
    variable_mean = np.zeros(variable.shape[0])
    variable_mean[:] = np.nan
    time = 0
    for time in range(variable.shape[0]):
        variable_mean[time] = np.average(variable_zonal[time,j_min:j_max+1],axis=0,weights=lat_weights[j_min:j_max+1])
    #
    return variable_mean




# This function is like the last one, but it works for masked arrays.
def spatial_mean_masked(variable,lats,j_min,j_max,i_min,i_max):
    print('Computing spatial mean. i='+str(i_min)+'-'+str(i_max)+', j='+str(j_min)+'-'+str(j_max)+'.  Points are inclusive.')
    variable_zonal = np.nanmean(variable[:,:,i_min:i_max+1],axis=2)
    lat_weights = np.cos(np.radians(lats))
    variable_mean = np.zeros(variable.shape[0])
    variable_mean[:] = np.nan
    time = 0
    for time in range(variable.shape[0]):
        variable_mean[time] = np.ma.average(variable_zonal[time,j_min:j_max+1],axis=0,weights=lat_weights[j_min:j_max+1])
    #
    return variable_mean


# This function loads a single variable from a file, as well as lat, lon, and time data
def load_spatial_data(experiment_dir,experiment_name,niters,variable_name,variable_type):
    #
    if variable_type == 'mean':
        variable = 'xam'
    #
    if variable_type == 'variance':
        variable = 'xav'
    #
    print("Loading "+variable_name+" ("+variable_type+") "+" data: " + experiment_dir + experiment_name)
    #
    file_handle = np.load(experiment_dir + experiment_name + '/r0/ensemble_'+variable_type+'_'+variable_name+'.npz')
    years = file_handle['years']
    lat = file_handle['lat']
    lon = file_handle['lon']
    nlat = file_handle['nlat']
    nlon = file_handle['nlon']
    file_handle.close()
    #
    # Initialize the data array as nan.
    data = np.zeros([niters,len(years),nlat,nlon])
    data[:] = np.nan
    #
    # Loop through the LMR members, saving each into an array.
    i = 0
    for i in range(niters):
        print(str(i+1) + "/" + str(niters))
        #
        file_handle = np.load(experiment_dir + experiment_name + '/r'+str(i)+'/ensemble_'+variable_type+'_'+variable_name+'.npz')
        data[i,:,:,:] = file_handle[variable]
        file_handle.close()
    #
    return years, lat, lon, data

