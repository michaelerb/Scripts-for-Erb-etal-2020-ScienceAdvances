#=============================================================
# This script computes a maximum covaiance analysis on scPDSI
# and other variables (tas, 500mb heights, or SLP).  It plots
# the first and second mode of the MCA.
#    authors:  Yuxin Zhou, Michael Erb
#    date   : 1/7/2019
#=============================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
from scipy import signal
import lmr_land_ocean_masks as masks
import xarray as xr
import copy
import statsmodels.api as sm

#===============
# USER SETTINGS
#===============

save_instead_of_plot = True

dataset = 'LMR'
#dataset = 'prior'
#iteration = 'mean'
iteration = sys.argv[1]

# If the variable should be an integer, make it an integer
try:    iteration = int(iteration)
except: print('Not an integer.')

# Only matters for LMR
experiment_name = 'production'
#experiment_name = 'NA_different'
#experiment_name = 'NA_different_annualPSM'

# Choose the variables to analyse
#variable_name1 = "precipitation"
variable_name1 = "PDSI"
variable_name2 = "tas_and_500mb_height"

# If desired, specify a lead for the Nino3.4 data.
var2_lead = 0
#var2_lead = 1

# Should the Nino3.4 or PDO be regressed off of the datasets?   WARNING: This doesn't work properly when setting a lead time above.
remove_influence_of_nino34 = False
remove_influence_of_pdo    = False
remove_influence_of_amo    = False

# Choose the principle component of interest
mode = 1

# Select the time of interest
if dataset == 'LMR':
    year_bounds = [1001,2000]
    #year_bounds = [1001,1850]
    #year_bounds = [1950,1997]
elif dataset == 'prior':
    year_bounds = [851,1850]

# regions of interest (lat min, lat max, lon min, lon max)
variable1_bounds = [ 25, 50, 230, 300]  # 25N-50N, 130W-60W  - ~US
#variable1_bounds = [ 32, 40, 235, 255]  # 32N-40N, 125W-105W - Southwest US
variable2_bounds = [-25, 70,  90, 355]  # 25S-70N,  90E-5E   - Pacific and Atlantic
#variable2_bounds = [-25, 70,  90, 260]  # 25S-70N,  90E-80W  - Pacific only
#variable2_bounds = [-25, 70, 260, 360]  # 25S-70N,  80W-5E   - Atlantic only

# Set this to 10 or 100 to compute decadal or centennial averages of variables.
averaging_period = 1
#averaging_period = 10

#===============


# Compute a decadal or centennial mean, if requested.
def decadal_or_centennial_means(variable_2d,averaging_period):
    #
    print("Computing "+str(averaging_period)+"-year means")
    variable_2d_mean = np.mean(np.reshape(variable_2d,(variable_2d.shape[0]/averaging_period,averaging_period,variable_2d.shape[1])),axis=1)
    return variable_2d_mean


### LOAD DATA

# The analysis doesn't work properly with certain combinations of parameters.  If those parameters are set, exit the script.
if (remove_influence_of_nino34 or remove_influence_of_pdo or remove_influence_of_amo) and var2_lead != 0:
    print('Removing the influence of a climate index does not work properly when setting a lead time.  Exiting.')
    exit()

### LOAD DATA
if dataset == 'LMR':
    #
    data_dir = '/projects/pd_lab/data/LMR/archive_output/'
    #
    # Sometimes, it's useful to do this analysis using two experiments.  If that's the case, specify them here.
    if experiment_name == 'production':
        experiment_name1 = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
        experiment_name2 = experiment_name1
    elif experiment_name == 'NA_different':
        experiment_name1 = 'NorthAmerica_only_r0_r4_temporary'
        experiment_name2 = 'NorthAmerica_exclude_r0_r4_temporary'
    elif experiment_name == 'NA_different_annualPSM':
        experiment_name1 = 'NorthAmerica_only_annualPSM_r0_r4_temporary'
        experiment_name2 = 'NorthAmerica_exclude_annualPSM_r0_r4_temporary'
    #
    handle = xr.open_dataset(data_dir+experiment_name1+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
    scpdsi_all = handle['pdsi'].values
    lon = handle['lon'].values
    lat = handle['lat'].values
    time = handle['time'].values
    handle.close()
    #
    #handle = xr.open_dataset(data_dir+experiment_name1+'/prate_MCruns_ensemble_mean.nc',decode_times=False)
    #pr_all = handle['prate'].values
    #handle.close()
    #
    handle = xr.open_dataset(data_dir+experiment_name2+'/sst_MCruns_ensemble_mean.nc',decode_times=False)
    sst_all = handle['sst'].values
    handle.close()
    #
    handle = xr.open_dataset(data_dir+experiment_name2+'/hgt500_MCruns_ensemble_mean.nc',decode_times=False)
    zg_500hPa_all = handle['hgt500'].values
    handle.close()
    #
    handle = xr.open_dataset(data_dir+experiment_name2+'/climate_indices_MCruns_ensemble_mean_calc_from_posterior.nc',decode_times=False)
    nino34_all = handle['nino34'].values
    soi_all    = handle['soi'].values
    pdo_all    = handle['pdo'].values
    amo_all    = handle['amo'].values
    handle.close()
    #
    years = time/365
    years = years.astype(int)
    #
    # Mean over all LMR iterations
    niterations = scpdsi_all.shape[1]
    if (isinstance(iteration,np.int)) and (0 <= iteration < niterations):
        scpdsi    = scpdsi_all[:,iteration,:,:]
        #pr        = pr_all[:,iteration,:,:]
        sst       = sst_all[:,iteration,:,:]
        zg_500hPa = zg_500hPa_all[:,iteration,:,:]
        nino34    = nino34_all[:,iteration]
        soi       = soi_all[:,iteration]
        pdo       = pdo_all[:,iteration]
        amo       = amo_all[:,iteration]
    else:
        iteration = 'mean'
        scpdsi    = np.mean(scpdsi_all,axis=1)
        #pr        = np.mean(pr_all,axis=1)
        sst       = np.mean(sst_all,axis=1)
        zg_500hPa = np.mean(zg_500hPa_all,axis=1)
        nino34    = np.mean(nino34_all,axis=1)
        soi       = np.mean(soi_all,axis=1)
        pdo       = np.mean(pdo_all,axis=1)
        amo       = np.mean(amo_all,axis=1)
    #
    # Mask the variables
    landmask, oceanmask = masks.masks(data_dir+experiment_name1+'/r0/ensemble_mean_tos_sfc_Omon.npz')
    scpdsi  = scpdsi*oceanmask[None,:,:]
    scpdsi  = ma.masked_invalid(scpdsi)
    #pr_land = pr*oceanmask[None,:,:]
    #pr_land = ma.masked_invalid(pr_land)
    sst     = ma.masked_invalid(sst)
    #
elif dataset == 'prior':
    #
    data_dir        = '/projects/pd_lab/data/LMR/data/model/ccsm4_last_millenium/'
    data_dir_regrid = '/projects/pd_lab/data/processed_data/LMR_regrid/data_regrid/'
    #
    handle = xr.open_dataset(data_dir+'scpdsipm_sfc_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
    scpdsi_all = handle['scpdsipm'].values
    lon = handle['lon'].values
    lat = handle['lat'].values
    time_bnds = handle['time_bnds'].values
    handle.close()
    #
    #handle = xr.open_dataset(data_dir+'pr_sfc_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
    #pr_all = handle['pr'].values
    #handle.close()
    #
    handle = xr.open_dataset(data_dir_regrid+'tos_sfc_Omon_CCSM4_past1000_085001-185012_regrid.nc',decode_times=False)
    sst_all = handle['tos'].values
    handle.close()
    #
    handle = xr.open_dataset(data_dir+'zg_500hPa_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
    zg_500hPa_all = handle['zg'].values
    zg_500hPa_all = np.squeeze(zg_500hPa_all)
    handle.close()
    #
    handle = xr.open_dataset(data_dir_regrid+'/climate_indices_CCSM4_past1000_085001-185012.nc',decode_times=False)
    nino34 = handle['nino34'].values
    soi    = handle['soi'].values
    pdo    = handle['pdo'].values
    amo    = handle['amo'].values
    handle.close()
    #
    # Compute annual-means
    def annual_mean(var,days_per_month):
        ntime = var.shape[0]
        nlat  = var.shape[1]
        nlon  = var.shape[2]
        nyears = ntime/12
        var_2d = np.reshape(var,(nyears,12,nlat,nlon))
        days_per_month_2d = np.reshape(days_per_month,(nyears,12))
        #
        var_annual = np.zeros((nyears,nlat,nlon)); var_annual[:] = np.nan
        for i in range(nyears):
            var_annual[i,:,:] = np.average(var_2d[i,:,:,:],axis=0,weights=days_per_month_2d[i,:])
        #
        return var_annual
    #
    days_per_month = time_bnds[:,1]-time_bnds[:,0]
    scpdsi    = annual_mean(scpdsi_all,   days_per_month)
    #pr        = annual_mean(pr_all,       days_per_month)
    sst       = annual_mean(sst_all,      days_per_month)
    zg_500hPa = annual_mean(zg_500hPa_all,days_per_month)
    #
    years = np.arange(850,1851)
    #
    # Mask the variables
    scpdsi[scpdsi == 0] = np.nan
    oceanmask = copy.deepcopy(scpdsi)
    oceanmask[np.isfinite(oceanmask)] = 1
    scpdsi  = ma.masked_invalid(scpdsi)
    #pr_land = pr*oceanmask
    #pr_land = ma.masked_invalid(pr_land)
    sst     = ma.masked_invalid(sst)

lon = np.append(lon,360) # Add the left-most lon point to the right, for later plotting purposes.
lon_2d,lat_2d = np.meshgrid(lon,lat)


### CALCULATIONS

# Change precipitation units to mm/day
#pr      = pr*60*60*24
#pr_land = pr_land*60*60*24

# Set the variables of interest
if variable_name1 == "precipitation":
    print('Precipitaiton has been chosen for variable1, but the code for precip has been commented out')
    print('to speed up the code.  It will not work as-is.')
    exit()
#    variable1 = pr_land
    units1 = "mm/day"
else:
    variable1 = scpdsi
    units1 = ""

if variable_name2 == "tas_and_500mb_height":
    variable2 = sst
    variable3 = zg_500hPa
    units2 = "deg C"
#    units3 = "m"

def find_midpoints(variable):
    variable_step1 = (variable[:-1,:] + variable[1:,:])/2
    variable_edges = (variable_step1[:,:-1] + variable_step1[:,1:])/2
    return variable_edges

# Select the region of interest for variable1
j_values_variable1 = np.where((lat >= variable1_bounds[0]) & (lat <= variable1_bounds[1]))[0].tolist()
i_values_variable1 = np.where((lon >= variable1_bounds[2]) & (lon <= variable1_bounds[3]))[0].tolist()
print("Region for variable 1: j="+str(j_values_variable1[0])+"-"+str(j_values_variable1[-1])+", i="+str(i_values_variable1[0])+"-"+str(i_values_variable1[-1]))
variable1_region = variable1[:,j_values_variable1[0]:j_values_variable1[-1]+1,i_values_variable1[0]:i_values_variable1[-1]+1]
lat_variable1    =      lat_2d[j_values_variable1[0]:j_values_variable1[-1]+1,i_values_variable1[0]:i_values_variable1[-1]+1]
lon_variable1    =      lon_2d[j_values_variable1[0]:j_values_variable1[-1]+1,i_values_variable1[0]:i_values_variable1[-1]+1]

# Select the region of interest for variable2
j_values_variable2 = np.where((lat >= variable2_bounds[0]) & (lat <= variable2_bounds[1]))[0].tolist()
i_values_variable2 = np.where((lon >= variable2_bounds[2]) & (lon <= variable2_bounds[3]))[0].tolist()
print("Region for variable 2: j="+str(j_values_variable2[0])+"-"+str(j_values_variable2[-1])+", i="+str(i_values_variable2[0])+"-"+str(i_values_variable2[-1]))
variable2_region = variable2[:,j_values_variable2[0]:j_values_variable2[-1]+1,i_values_variable2[0]:i_values_variable2[-1]+1]
variable3_region = variable3[:,j_values_variable2[0]:j_values_variable2[-1]+1,i_values_variable2[0]:i_values_variable2[-1]+1]
lat_variable2    =      lat_2d[j_values_variable2[0]:j_values_variable2[-1]+1,i_values_variable2[0]:i_values_variable2[-1]+1]
lon_variable2    =      lon_2d[j_values_variable2[0]:j_values_variable2[-1]+1,i_values_variable2[0]:i_values_variable2[-1]+1]

# Compute the verticies of the LMR grid, for plotting purposes
lat_more_variable1 = lat_2d[j_values_variable1[0]-1:j_values_variable1[-1]+2,i_values_variable1[0]-1:i_values_variable1[-1]+2]
lon_more_variable1 = lon_2d[j_values_variable1[0]-1:j_values_variable1[-1]+2,i_values_variable1[0]-1:i_values_variable1[-1]+2]
lat_more_variable2 = lat_2d[j_values_variable2[0]-1:j_values_variable2[-1]+2,i_values_variable2[0]-1:i_values_variable2[-1]+2]
lon_more_variable2 = lon_2d[j_values_variable2[0]-1:j_values_variable2[-1]+2,i_values_variable2[0]-1:i_values_variable2[-1]+2]
lat_edges_variable1 = find_midpoints(lat_more_variable1)
lon_edges_variable1 = find_midpoints(lon_more_variable1)
lat_edges_variable2 = find_midpoints(lat_more_variable2)
lon_edges_variable2 = find_midpoints(lon_more_variable2)

# Select the time of interest
def shorten_data(data,years,yearmin,yearmax):
    indexmin = np.where(years == yearmin)[0][0]
    indexmax = np.where(years == yearmax)[0][0]
    if len(data.shape) == 3:
        data_new = data[indexmin:indexmax+1,:,:]
    else:
        data_new = data[indexmin:indexmax+1]
    return data_new

variable1_selected = shorten_data(variable1_region,years,year_bounds[0],          year_bounds[1])
variable2_selected = shorten_data(variable2_region,years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
variable3_selected = shorten_data(variable3_region,years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
nino34_selected    = shorten_data(nino34,          years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
#soi_selected       = shorten_data(soi,             years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
pdo_selected       = shorten_data(pdo,             years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
amo_selected       = shorten_data(amo,             years,year_bounds[0]-var2_lead,year_bounds[1]-var2_lead)
years_selected     = shorten_data(years,           years,year_bounds[0],          year_bounds[1])

# Remove the time-mean from each grid point in all data
variable1_selected = variable1_selected - np.mean(variable1_selected,axis=0)
variable2_selected = variable2_selected - np.mean(variable2_selected,axis=0)
variable3_selected = variable3_selected - np.mean(variable3_selected,axis=0)

# If desired, compute a regression between a time series and other climate variables at every point,
# then remove the calculated patterns from the climate fields.
def remove_regression(variable,ts_to_regress):
    nlat = variable.shape[1]
    nlon = variable.shape[2]
    #
    slopes     = np.zeros((nlat,nlon)); slopes[:]     = np.nan
    intercepts = np.zeros((nlat,nlon)); intercepts[:] = np.nan
    #
    ts_to_regress_ac = sm.add_constant(ts_to_regress)
    for j in range(nlat):
        print('Removing regression: '+str(j+1)+'/'+str(nlat))
        for i in range(nlon):
            intercepts[j,i],slopes[j,i] = sm.OLS(variable[:,j,i],ts_to_regress_ac).fit().params
    #
    variable_regress_removed = variable - ((ts_to_regress[:,None,None]*slopes[None,:,:]) + intercepts[None,:,:])
    return variable_regress_removed

if remove_influence_of_nino34 == True:
    variable1_selected = remove_regression(variable1_selected,nino34_selected)
    variable2_selected = remove_regression(variable2_selected,nino34_selected)
    variable3_selected = remove_regression(variable3_selected,nino34_selected)
if remove_influence_of_pdo == True:
    variable1_selected = remove_regression(variable1_selected,pdo_selected)
    variable2_selected = remove_regression(variable2_selected,pdo_selected)
    variable3_selected = remove_regression(variable3_selected,pdo_selected)
if remove_influence_of_amo == True:
    variable1_selected = remove_regression(variable1_selected,amo_selected)
    variable2_selected = remove_regression(variable2_selected,amo_selected)
    variable3_selected = remove_regression(variable3_selected,amo_selected)


# Reshape to 2d
nt = variable1_selected.shape[0]
nlat_variable1 = variable1_selected.shape[1]
nlon_variable1 = variable1_selected.shape[2]
nlat_variable2 = variable2_selected.shape[1]
nlon_variable2 = variable2_selected.shape[2]
variable1_2d = np.reshape(variable1_selected, (nt, nlat_variable1*nlon_variable1))
variable2_2d = np.reshape(variable2_selected, (nt, nlat_variable2*nlon_variable2))
variable3_2d = np.reshape(variable3_selected, (nt, nlat_variable2*nlon_variable2))

# Standardize data to a mean of 0 and a standard deviation of 1:
def standardize_data(variable):
    variable_mean = np.mean(variable.flatten())
    variable_std  = np.std(variable.flatten())
    variable = (variable - variable_mean)/variable_std
    return variable

variable1_2d = standardize_data(variable1_2d)
variable2_2d = standardize_data(variable2_2d)
variable3_2d = standardize_data(variable3_2d)

# If specified, average the variable into decadal and centenial averages.
if (averaging_period == 10) or (averaging_period == 100):
    variable1_2d = decadal_or_centennial_means(variable1_2d,averaging_period)
    variable2_2d = decadal_or_centennial_means(variable2_2d,averaging_period)
    variable3_2d = decadal_or_centennial_means(variable3_2d,averaging_period)
    years_selected = np.mean(np.reshape(years_selected,(years_selected.shape[0]/averaging_period,averaging_period)),axis=1)

# convert nan to zero in the arrays as the way to process missing data
variable1_2d = np.nan_to_num(variable1_2d)
variable2_2d = np.nan_to_num(variable2_2d)
variable3_2d = np.nan_to_num(variable3_2d)

# Join variables 2 and 3
variable2_2d = np.concatenate((variable2_2d,variable3_2d),axis=1)


mode = 1


# Function to compute MCA for variable1 vs variable2
def mca(variable1_2d,variable2_2d,years_selected,nt,mode,lat_variable1,lon_variable1,lat_variable2):
    #
    # Define number of lats and lons
    nlat_variable1 = lat_variable1.shape[0]
    nlon_variable1 = lat_variable1.shape[1]
    nlat_variable2 = lat_variable2.shape[0]
    nlon_variable2 = lat_variable2.shape[1]
    #
    # covariance matrix
    cov = np.dot(variable1_2d.T, variable2_2d.T.T) / (nt-1) # note that from this line on variable1 is in front of variable2
    # In Bretherton et al. [1992] the spatial-temperal data has gridpoint as the first dimension and time as the second dimension, which is the opposite as mine. Thus the transpose of the two matrix before computing covariance matrix.
    #
    # svd
    # Single Value Decomposition.  U (the left singular vectors) is like the EOFs.  V (the right singular vectors) is like the PCs.
    U, s, V = np.linalg.svd(cov)
    #
    # scf - squared covariance fraction
    scf = 100 * s[mode-1]**2 / np.sum(i**2 for i in s)
    #
    # fov - fraction of variance
    variable1_ec_var = []
    for i in range(len(U)):
        variable1_ec_var.append(np.var(np.dot(U[:,i].T, variable1_2d.T)))
    #
    #variable1_fov = max(variable1_ec_var) / sum(variable1_ec_var)
    variable1_fov = variable1_ec_var[mode-1] / sum(variable1_ec_var)
    #
    variable2_ec_var = []
    for i in range(len(V)):
        variable2_ec_var.append(np.var(np.dot(V[i,:].T, variable2_2d.T)))
    #
    #variable2_fov = max(variable2_ec_var) / sum(variable2_ec_var)
    variable2_fov = variable2_ec_var[mode-1] / sum(variable2_ec_var)
    #
    # expansion coefficients (similar to principle components for the first EOF)
    variable1_ec_1 = np.dot(U[:,mode-1].T, variable1_2d.T)
    variable2_ec_1 = np.dot(V[mode-1,:].T, variable2_2d.T)
    #
    # standardize expansion coefficients and compute their correlation coefficient
    variable1_ec_1_n = (variable1_ec_1 - np.mean(variable1_ec_1) * np.ones(len(variable1_ec_1))) / np.std(variable1_ec_1)  # normalized
    variable2_ec_1_n = (variable2_ec_1 - np.mean(variable2_ec_1) * np.ones(len(variable2_ec_1))) / np.std(variable2_ec_1)  # normalized
    corrcoef = np.corrcoef(variable1_ec_1_n, variable2_ec_1_n)[0,1]
    #
    # homogeneous "covariance" map for variable1
    variable1_homo_1d = np.dot(variable1_2d.T, variable1_ec_1_n) / (nt-1)
    variable1_homo = np.reshape(variable1_homo_1d, (nlat_variable1, nlon_variable1))
    #
    variable1_hete_1d = np.dot(variable1_2d.T, variable2_ec_1_n) / (nt-1)
    variable1_hete = np.reshape(variable1_hete_1d, (nlat_variable1, nlon_variable1))
    # see above for the reason of the transpose of variable1_2d and variable2_2d
    #
    # heterogeneous "covariance" map for variable2
    variable2_homo_1d = np.dot(variable2_2d.T, variable2_ec_1_n.T) / (nt-1)
    variable2_hete_1d = np.dot(variable2_2d.T, variable1_ec_1_n.T) / (nt-1)
    #
    # Split variables 2 and 3 apart again.
    variable2_homo_1d, variable3_homo_1d = np.split(variable2_homo_1d,2,axis=0)
    variable2_hete_1d, variable3_hete_1d = np.split(variable2_hete_1d,2,axis=0)
    #
    # Reshape the variables into 2d
    variable2_homo = np.reshape(variable2_homo_1d, (nlat_variable2, nlon_variable2))
    variable3_homo = np.reshape(variable3_homo_1d, (nlat_variable2, nlon_variable2))
    variable2_hete = np.reshape(variable2_hete_1d, (nlat_variable2, nlon_variable2))
    variable3_hete = np.reshape(variable3_hete_1d, (nlat_variable2, nlon_variable2))
    #
    """
    plt.contourf(lat_variable1)
    plt.colorbar()
    plt.contourf(variable1_homo)
    plt.colorbar()
    #
    lats_flatten           = np.reshape(lat_variable1, (nlat_variable1*nlon_variable1))
    variable1_homo_flatten = np.reshape(variable1_homo,(nlat_variable1*nlon_variable1))
    # Find only valid variables
    indices_valid = np.isfinite(variable1_homo_flatten)
    lats_valid           = lats_flatten[indices_valid]
    variable1_homo_valid = variable1_homo_flatten[indices_valid]
    lat_weights = np.cos(np.radians(lats_valid))
    variable1_homo_mean = np.average(variable1_homo_valid,weights=lat_weights)
    """
    #
    # Make sure that the MCA maps correspond to mean drought in the U.S.
    variable1_homo_zonal = np.nanmean(variable1_homo,axis=1)
    lat_weights = np.cos(np.radians(lat_variable1[:,0]))
    variable1_homo_mean = np.average(variable1_homo_zonal,axis=0,weights=lat_weights)
    if variable1_homo_mean > 0:
        variable1_ec_1    = -1*variable1_ec_1
        variable2_ec_1    = -1*variable2_ec_1
        variable1_ec_1_n  = -1*variable1_ec_1_n
        variable2_ec_1_n  = -1*variable2_ec_1_n
        variable1_homo_1d = -1*variable1_homo_1d
        variable1_homo    = -1*variable1_homo
        variable1_hete_1d = -1*variable1_hete_1d
        variable1_hete    = -1*variable1_hete
        variable2_homo_1d = -1*variable2_homo_1d
        variable2_homo    = -1*variable2_homo
        variable2_hete_1d = -1*variable2_hete_1d
        variable2_hete    = -1*variable2_hete
        variable3_homo_1d = -1*variable3_homo_1d
        variable3_homo    = -1*variable3_homo
        variable3_hete_1d = -1*variable3_hete_1d
        variable3_hete    = -1*variable3_hete
    #
    # Package the outputs, then return necessary variables for figures
    mca_outputs = {}
    mca_outputs['variable1_hete']      = variable1_hete
    mca_outputs['variable2_hete']      = variable2_hete
    mca_outputs['variable3_hete']      = variable3_hete
    mca_outputs['variable1_fov']       = variable1_fov
    mca_outputs['variable2_fov']       = variable2_fov
    mca_outputs['scf']                 = scf
    mca_outputs['variable1_ec_1_n']    = variable1_ec_1_n
    mca_outputs['variable2_ec_1_n']    = variable2_ec_1_n
    mca_outputs['corrcoef']            = corrcoef
    mca_outputs['lon_edges_variable1'] = lon_edges_variable1
    mca_outputs['lat_edges_variable1'] = lat_edges_variable1
    mca_outputs['lon_edges_variable2'] = lon_edges_variable2
    mca_outputs['lat_edges_variable2'] = lat_edges_variable2
    mca_outputs['lon_variable2']       = lon_variable2
    mca_outputs['lat_variable2']       = lat_variable2
    mca_outputs['units1']              = units1
    mca_outputs['units2']              = units2
    mca_outputs['years_selected']      = years_selected
    return mca_outputs


# Make figures in the original way
def make_figures_original(mca_outputs,iteration,save_instead_of_plot):
    #
    ### FIGURES
    plt.style.use('ggplot')
    #
    # FIGURE 1: plot variable1_hete and variable2_hete
    plt.figure(figsize=(19,8))
    #
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=variable1_bounds[0],urcrnrlat=variable1_bounds[1],llcrnrlon=variable1_bounds[2],urcrnrlon=variable1_bounds[3],resolution='c')
    x_edges_variable1, y_edges_variable1 = m(mca_outputs['lon_edges_variable1'], mca_outputs['lat_edges_variable1'])
    #
    plt.axes([.025,.025,.45,.9])
    m.pcolormesh(x_edges_variable1, y_edges_variable1, mca_outputs['variable1_hete'], cmap='BrBG', vmin = -1*np.abs(mca_outputs['variable1_hete']).max(), vmax = np.abs(mca_outputs['variable1_hete']).max())
    m.drawcoastlines()
    m.drawparallels(np.arange(-80,90,5),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1])
    cb = plt.colorbar(orientation='horizontal')
    cb.set_label(variable_name1+' anomaly ('+mca_outputs['units1']+')', fontsize=12)
    plt.title('a) Heterogeneous covariance map of '+variable_name1+' Mode '+str(mode)+'\n(Squared covariance fraction = ' + str(int(mca_outputs['scf'])) + '%, Fraction of variance = ' + str(int(mca_outputs['variable1_fov']*100)), fontsize=16)
    #
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=variable2_bounds[0],urcrnrlat=variable2_bounds[1],llcrnrlon=variable2_bounds[2],urcrnrlon=variable2_bounds[3],resolution='c')
    x_variable2, y_variable2 = m(mca_outputs['lon_variable2'], mca_outputs['lat_variable2'])
    x_edges_variable2, y_edges_variable2 = m(mca_outputs['lon_edges_variable2'], mca_outputs['lat_edges_variable2'])
    #
    plt.axes([.525,.025,.45,.9])
    filled_contours = m.pcolormesh(x_edges_variable2, y_edges_variable2, mca_outputs['variable2_hete'], cmap='RdBu_r', vmin = -1*np.abs(mca_outputs['variable2_hete']).max(), vmax = np.abs(mca_outputs['variable2_hete']).max())
    m.contour(x_variable2, y_variable2, mca_outputs['variable3_hete'], 20, colors='k', linewidths=1, vmin = -1*np.abs(mca_outputs['variable3_hete']).max(), vmax = np.abs(mca_outputs['variable3_hete']).max())
    m.drawcoastlines()
    m.drawparallels(np.arange(-80,90,20),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0,360,20),labels=[0,0,0,1])
    cb = plt.colorbar(filled_contours,orientation='horizontal')
    cb.set_label(variable_name2+' anomaly ('+mca_outputs['units2']+')', fontsize=12)
    plt.title('b) Heterogeneous covariance map of '+variable_name2+' Mode '+str(mode)+', '+dataset+'\n(Fraction of variance = ' + str(round(mca_outputs['variable2_fov'],2)*100) + '%). Lead: '+str(var2_lead)+' year', fontsize=16)
    #
    if save_instead_of_plot:
        plt.savefig("figures/mca_"+dataset+"_"+variable_name1+"_"+variable_name2+"_PC_"+str(mode)+"_years_"+str(year_bounds[0])+"_"+str(year_bounds[1])+"_"+str(averaging_period)+"_yr_mean_"+str(remove_influence_of_nino34)+"_"+str(remove_influence_of_pdo)+"_"+str(remove_influence_of_amo)+"_lead_"+str(var2_lead)+"_iter"+str(iteration)+".png",dpi=300,format='png')
        plt.close()
    else:
        plt.show()
    #
    #
    # # apply low pass filter on them
    N = 2
    Wn = 0.1
    b, a = signal.butter(N, Wn)
    variable1_ec_1_n_f = signal.filtfilt(b, a, mca_outputs['variable1_ec_1_n'])
    variable2_ec_1_n_f = signal.filtfilt(b, a, mca_outputs['variable2_ec_1_n'])
    #
    # plot standardized expasion coefficients
    plt.figure(figsize=(12,5))
    plt.clf()
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    p1, = plt.plot(mca_outputs['variable1_ec_1_n'], color='#0099FF', linewidth=1)
    p2, = plt.plot(mca_outputs['variable2_ec_1_n'], color='#FF6600', linewidth=1)
    p3, = plt.plot(variable1_ec_1_n_f, color = '#0000FF', linewidth=2)
    p4, = plt.plot(variable2_ec_1_n_f, color = '#FF0000', linewidth=2)
    plt.title('Standardized expansion coefficients, '+dataset+'\nRemove Nino3.4: '+str(remove_influence_of_nino34)+', remove PDO: '+str(remove_influence_of_pdo)+', remove AMO: '+str(remove_influence_of_amo))
    plt.figtext(0.92,0.88,'r=' + str(round(mca_outputs['corrcoef'],2)))
    plt.xlabel('Time')
    plt.ylabel('Coefficient (unitless)')
    plt.legend([p1, p2, p3, p4], ["First left expansion coefficient ("+variable_name1+")", "First right expansion coefficient ("+variable_name2+")", 'Filtered left e.c.', 'Filter right e.c.'], loc=3, fontsize=12)
    plt.ylim(-4,4)
    plt.tight_layout()
    if save_instead_of_plot:
        plt.savefig("figures/mca_ec_"+dataset+"_"+variable_name1+"_"+variable_name2+"_PC_"+str(mode)+"_years_"+str(year_bounds[0])+"_"+str(year_bounds[1])+"_"+str(averaging_period)+"_yr_mean_"+str(remove_influence_of_nino34)+"_"+str(remove_influence_of_pdo)+"_"+str(remove_influence_of_amo)+"_lead"+str(var2_lead)+"_iter_"+str(iteration)+".png",dpi=300,format='png')
        plt.close()
    else:
        plt.show()

#mca_outputs1,mca_outputs2,mode1,mode2 = mca_outputs_mode_1,mca_outputs_mode_2,1,2

# Make figures formatted for the paper.
def make_figure_for_paper(mca_outputs1,mca_outputs2,mode1,mode2,iteration,save_instead_of_plot):
    #
    ### FIGURES
    plt.style.use('ggplot')
    #
    mca_max_1 = np.max([ np.abs(mca_outputs1['variable1_hete']).max(), np.abs(mca_outputs2['variable1_hete']).max() ])
    mca_max_2 = np.max([ np.abs(mca_outputs1['variable2_hete']).max(), np.abs(mca_outputs2['variable2_hete']).max() ])
    mca_max_3 = np.max([ np.abs(mca_outputs1['variable3_hete']).max(), np.abs(mca_outputs2['variable3_hete']).max() ])
    #
    f, ax = plt.subplots(2,2,figsize=(18,9))
    ax = ax.ravel()
    #
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=variable1_bounds[0],urcrnrlat=variable1_bounds[1],llcrnrlon=variable1_bounds[2],urcrnrlon=variable1_bounds[3],resolution='c')
    x_edges_variable1, y_edges_variable1 = m(mca_outputs1['lon_edges_variable1'], mca_outputs1['lat_edges_variable1'])
    #
    pdsi_map = m.pcolormesh(x_edges_variable1, y_edges_variable1, mca_outputs1['variable1_hete'], cmap='BrBG', vmin = -1*mca_max_1, vmax = mca_max_1,ax=ax[0])
    m.drawcoastlines(ax=ax[0])
    m.drawparallels(np.arange(-80,90,5),labels=[1,0,0,0],ax=ax[0])
    m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],ax=ax[0])
    cb = m.colorbar(pdsi_map,ax=ax[0])
    cb.set_label('PDSI anomaly', fontsize=14)
    plt.text(1,.10,'SCF=' + str('%1.2f' % (mca_outputs1['scf']/100)),    transform=ax[0].transAxes,horizontalalignment='right',fontsize=14)
    plt.text(1,.03,'FOV=' + str('%1.2f' % mca_outputs1['variable1_fov']),transform=ax[0].transAxes,horizontalalignment='right',fontsize=14)
    ax[0].set_title('a) PDSI, mode '+str(mode1),fontsize=18,loc='left')
    #
    pdsi_map = m.pcolormesh(x_edges_variable1, y_edges_variable1, mca_outputs2['variable1_hete'], cmap='BrBG', vmin = -1*mca_max_1, vmax = mca_max_1,ax=ax[2])
    m.drawcoastlines(ax=ax[2])
    m.drawparallels(np.arange(-80,90,5),labels=[1,0,0,0],ax=ax[2])
    m.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1],ax=ax[2])
    cb = m.colorbar(pdsi_map,ax=ax[2])
    cb.set_label('PDSI anomaly', fontsize=14)
    plt.text(1,.10,'SCF=' + str('%1.2f' % (mca_outputs2['scf']/100)),    transform=ax[2].transAxes,horizontalalignment='right',fontsize=14)
    plt.text(1,.03,'FOV=' + str('%1.2f' % mca_outputs2['variable1_fov']),transform=ax[2].transAxes,horizontalalignment='right',fontsize=14)
    ax[2].set_title('c) PDSI, mode '+str(mode2),fontsize=18,loc='left')
    #
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=variable2_bounds[0],urcrnrlat=variable2_bounds[1],llcrnrlon=variable2_bounds[2],urcrnrlon=variable2_bounds[3],resolution='c')
    x_variable2, y_variable2 = m(mca_outputs1['lon_variable2'], mca_outputs1['lat_variable2'])
    x_edges_variable2, y_edges_variable2 = m(mca_outputs1['lon_edges_variable2'], mca_outputs1['lat_edges_variable2'])
    #
    filled_contours = m.pcolormesh(x_edges_variable2, y_edges_variable2, mca_outputs1['variable2_hete'], cmap='RdBu_r', vmin = -1*mca_max_2, vmax = mca_max_2,ax=ax[1])
    m.contour(x_variable2, y_variable2, mca_outputs1['variable3_hete'], 10, colors='k', linewidths=1, vmin = -1*mca_max_3, vmax = mca_max_3,ax=ax[1])
    m.drawcoastlines(ax=ax[1])
    m.drawparallels(np.arange(-80,90,20),labels=[1,0,0,0],ax=ax[1])
    m.drawmeridians(np.arange(0,360,40),labels=[0,0,0,1],ax=ax[1])
    cb = m.colorbar(filled_contours,ax=ax[1])
    cb.set_label('Temperature', fontsize=14)
    plt.text(1,.06,'FOV=' + str('%1.2f' % mca_outputs1['variable2_fov']),transform=ax[1].transAxes,horizontalalignment='right',fontsize=14)
    ax[1].set_title('b) Temperature and 500 hPa heights, mode '+str(mode1),fontsize=18,loc='left')
    #
    filled_contours = m.pcolormesh(x_edges_variable2, y_edges_variable2, mca_outputs2['variable2_hete'], cmap='RdBu_r', vmin = -1*mca_max_2, vmax = mca_max_2,ax=ax[3])
    m.contour(x_variable2, y_variable2, mca_outputs2['variable3_hete'], 10, colors='k', linewidths=1, vmin = -1*mca_max_3, vmax = mca_max_3,ax=ax[3])
    m.drawcoastlines(ax=ax[3])
    m.drawparallels(np.arange(-80,90,20),labels=[1,0,0,0],ax=ax[3])
    m.drawmeridians(np.arange(0,360,40),labels=[0,0,0,1],ax=ax[3])
    cb = m.colorbar(filled_contours,ax=ax[3])
    cb.set_label('Temperature', fontsize=14)
    plt.text(1,.06,'FOV=' + str('%1.2f' % mca_outputs2['variable2_fov']),transform=ax[3].transAxes,horizontalalignment='right',fontsize=14)
    ax[3].set_title('d) Temperature and 500 hPa heights, mode '+str(mode2),fontsize=18,loc='left')
    #
    if save_instead_of_plot:
        plt.savefig("figures/paper_mca_"+dataset+"_"+variable_name1+"_"+variable_name2+"_PC_"+str(mode1)+"_"+str(mode2)+"_years_"+str(year_bounds[0])+"_"+str(year_bounds[1])+"_"+str(averaging_period)+"_yr_mean_"+str(remove_influence_of_nino34)+"_"+str(remove_influence_of_pdo)+"_"+str(remove_influence_of_amo)+"_lead"+str(var2_lead)+"_iter_"+str(iteration)+".png",dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    #
    #
    # Plot the expansion coefficients
    f, ax = plt.subplots(2,1,figsize=(14,9))
    ax = ax.ravel()
    #
    ec_max = np.max([ np.abs(mca_outputs1['variable1_ec_1_n']).max(), np.abs(mca_outputs1['variable2_ec_1_n']).max(), np.abs(mca_outputs2['variable1_ec_1_n']).max(), np.abs(mca_outputs2['variable2_ec_1_n']).max() ])
    ec_max = np.ceil(ec_max)
    #
    N = 2
    Wn = 0.1
    b, a = signal.butter(N, Wn)
    variable1_ec_1_n_f_1 = signal.filtfilt(b, a, mca_outputs1['variable1_ec_1_n'])
    variable2_ec_1_n_f_1 = signal.filtfilt(b, a, mca_outputs1['variable2_ec_1_n'])
    variable1_ec_1_n_f_2 = signal.filtfilt(b, a, mca_outputs2['variable1_ec_1_n'])
    variable2_ec_1_n_f_2 = signal.filtfilt(b, a, mca_outputs2['variable2_ec_1_n'])
    years_selected_1_f   = signal.filtfilt(b, a, mca_outputs1['years_selected'])
    years_selected_2_f   = signal.filtfilt(b, a, mca_outputs2['years_selected'])
    #
    # plot standardized expasion coefficients, mode 1
    p1, = ax[0].plot(mca_outputs1['years_selected'],mca_outputs1['variable1_ec_1_n'], color='#0099FF', linewidth=1)
    p2, = ax[0].plot(mca_outputs1['years_selected'],mca_outputs1['variable2_ec_1_n'], color='#FF6600', linewidth=1)
    p3, = ax[0].plot(years_selected_1_f,variable1_ec_1_n_f_1, color = '#0000FF', linewidth=2)
    p4, = ax[0].plot(years_selected_1_f,variable2_ec_1_n_f_1, color = '#FF0000', linewidth=2)
    ax[0].axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax[0].set_title('a) Standardized expansion coefficients, mode 1',fontsize=16,loc='left')
    plt.text(.93,.92,'r=' + str('%1.2f' % mca_outputs1['corrcoef']),transform=ax[0].transAxes,fontsize=14)
#    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Coefficient (unitless)', fontsize=14)
    ax[0].set_xlim(mca_outputs1['years_selected'][0],mca_outputs1['years_selected'][-1])
    ax[0].set_ylim(-1*ec_max,ec_max)
    #
    # plot standardized expasion coefficients, mode 1
    p1, = ax[1].plot(mca_outputs2['years_selected'],mca_outputs2['variable1_ec_1_n'], color='#0099FF', linewidth=1)
    p2, = ax[1].plot(mca_outputs2['years_selected'],mca_outputs2['variable2_ec_1_n'], color='#FF6600', linewidth=1)
    p3, = ax[1].plot(years_selected_2_f,variable1_ec_1_n_f_2, color = '#0000FF', linewidth=2)
    p4, = ax[1].plot(years_selected_2_f,variable2_ec_1_n_f_2, color = '#FF0000', linewidth=2)
    ax[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax[1].set_title('b) Standardized expansion coefficients, mode 2',fontsize=16,loc='left')
    plt.text(.93,.92,'r=' + str('%1.2f' % mca_outputs2['corrcoef']),transform=ax[1].transAxes,fontsize=14)
    ax[1].set_xlabel('Time', fontsize=14)
    ax[1].set_ylabel('Coefficient (unitless)', fontsize=14)
    ax[1].legend([p1, p2, p3, p4], ["Expansion coefficient, PDSI", "Expansion coefficient, T and 500 hPa heights", 'Filtered E.C., PDSI', 'Filtered E.C., T and 500 hPa heights'],ncol=2,fontsize=12,loc=1,bbox_to_anchor=(1,-.14),prop={'size':11})
    ax[1].set_xlim(mca_outputs2['years_selected'][0],mca_outputs2['years_selected'][-1])
    ax[1].set_ylim(-1*ec_max,ec_max)
    #
    if save_instead_of_plot:
        plt.savefig("figures/mca_ec_"+dataset+"_"+variable_name1+"_"+variable_name2+"_PC_"+str(mode1)+"_"+str(mode2)+"_years_"+str(year_bounds[0])+"_"+str(year_bounds[1])+"_"+str(averaging_period)+"_yr_mean_"+str(remove_influence_of_nino34)+"_"+str(remove_influence_of_pdo)+"_"+str(remove_influence_of_amo)+"_lead"+str(var2_lead)+"_iter_"+str(iteration)+".png",dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Make a function to save the outputs
def save_outputs(mca_outputs1,mca_outputs2,mode1,mode2,iteration):
    #
    # Save the reconstruction to an output file.
    output_file = 'data/data_mca_'+dataset+'_'+variable_name1+'_'+variable_name2+'_PC_'+str(mode1)+'_'+str(mode2)+'_years_'+str(year_bounds[0])+'_'+str(year_bounds[1])+'_'+str(averaging_period)+'_yr_mean_'+str(remove_influence_of_nino34)+'_'+str(remove_influence_of_pdo)+'_'+str(remove_influence_of_amo)+'_lead'+str(var2_lead)+'_iter_'+str(iteration)+'.npz'
    np.savez(output_file,mca_outputs1=mca_outputs1,mca_outputs2=mca_outputs2,mode1=mode1,mode2=mode2,iteration=iteration)



mca_outputs_mode_1 = mca(variable1_2d,variable2_2d,years_selected,nt,1,lat_variable1,lon_variable1,lat_variable2)
mca_outputs_mode_2 = mca(variable1_2d,variable2_2d,years_selected,nt,2,lat_variable1,lon_variable1,lat_variable2)
make_figure_for_paper(mca_outputs_mode_1,mca_outputs_mode_2,1,2,iteration,save_instead_of_plot)
save_outputs(mca_outputs_mode_1,mca_outputs_mode_2,1,2,iteration)

make_figures_original(mca_outputs_mode_1,1,save_instead_of_plot)
make_figures_original(mca_outputs_mode_2,2,save_instead_of_plot)


"""
# Do the MCA calculations
mca_outputs_mode_3 = mca(variable1_2d,variable2_2d,years_selected,nt,3,lat_variable1,lon_variable1,lat_variable2)
mca_outputs_mode_4 = mca(variable1_2d,variable2_2d,years_selected,nt,4,lat_variable1,lon_variable1,lat_variable2)
make_figure_for_paper(mca_outputs_mode_3,mca_outputs_mode_4,3,4,iteration,save_instead_of_plot)
"""

