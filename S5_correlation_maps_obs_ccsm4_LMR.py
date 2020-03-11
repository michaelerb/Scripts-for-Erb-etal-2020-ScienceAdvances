#=============================================================================
# This script compares Nino3.4 (over multiple time periods) against PDSI at
# every U.S. location in three data sets:
#    1) Observational data
#    2) CCSM4 prior
#    3) LMR posterior
#    author: Michael P. Erb
#    date  : 7/25/2018
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import scipy.io
import mpe_functions as mpe
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import compute_regional_means
from mpl_toolkits import basemap


save_instead_of_plot = True

# Specify the regional bounds
region_bounds = {}
region_bounds['Southwest US']     = [32,  40,-125,  -105]  # 32-40N, 125-105W
region_bounds['Southwest US new'] = [28,  40,-115,   -95]  # 28-40N, 115-95W
possible_regions = ['Southwest US','Southwest US new']


### LOAD DATA

# Load LMR data
lmr_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'

handle = xr.open_dataset(lmr_dir+experiment_name+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_lmr_all = handle['pdsi'].values
lat_lmr        = handle['lat'].values
lon_lmr        = handle['lon'].values
time_lmr       = handle['time'].values
handle.close()

handle = xr.open_dataset(lmr_dir+experiment_name+'/climate_indices_MCruns_ensemble_mean_calc_from_posterior.nc',decode_times=False)
nino34_lmr_all = handle['nino34'].values
handle.close()

years_lmr = time_lmr/365
years_lmr = years_lmr.astype(int)


# Load CCSM4 prior
ccsm4_dir        = '/projects/pd_lab/data/LMR/data/model/ccsm4_last_millenium/'
ccsm4_dir_regrid = '/projects/pd_lab/data/processed_data/LMR_regrid/data_regrid/'

handle = xr.open_dataset(ccsm4_dir+'scpdsipm_sfc_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
scpdsi_ccsm4_all = handle['scpdsipm'].values
lon_ccsm4        = handle['lon'].values
lat_ccsm4        = handle['lat'].values
time_bnds_ccsm4  = handle['time_bnds'].values
handle.close()

handle = xr.open_dataset(ccsm4_dir_regrid+'/climate_indices_CCSM4_past1000_085001-185012.nc',decode_times=False)
nino34_ccsm4_mean = handle['nino34'].values
handle.close()

# Compute annual-means
#var,days_per_month = scpdsi_ccsm4_all,days_per_month_ccsm4
def annual_mean(var,days_per_month):
    ntime = var.shape[0]
    nlat  = var.shape[1]
    nlon  = var.shape[2]
    nyears = int(ntime/12)
    var_2d = np.reshape(var,(nyears,12,nlat,nlon))
    days_per_month_2d = np.reshape(days_per_month,(nyears,12))
    #
    var_annual = np.zeros((nyears,nlat,nlon)); var_annual[:] = np.nan
    for i in range(nyears):
        var_annual[i,:,:] = np.average(var_2d[i,:,:,:],axis=0,weights=days_per_month_2d[i,:])
    #
    return var_annual

days_per_month_ccsm4 = time_bnds_ccsm4[:,1]-time_bnds_ccsm4[:,0]
scpdsi_ccsm4_mean    = annual_mean(scpdsi_ccsm4_all,days_per_month_ccsm4)

years_ccsm4 = np.arange(850,1851)


# Load observational data

# DaiPDSI data set
handle_dai = xr.open_dataset('/projects/pd_lab/data/LMR/data/analyses/DaiPDSI/Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc',decode_times=False)
pdsi_dai_monthly = handle_dai['pdsi'].values
lat_dai          = handle_dai['lat'].values
lon_dai          = handle_dai['lon'].values
handle_dai.close()
years_dai = np.arange(1850,2015)

# Bunge & Clark Nino3.4 data - Data runs from January 1873 to March 2008
nino34_obs = scipy.io.loadmat('/projects/pd_lab/data/modern_datasets/climate_indices/Nino34/Bunge_and_Clarke_2009/NINO34.mat')
nino34_monthly = np.squeeze(nino34_obs['nino34'])
nino34_ndays = np.diff(nino34_obs['tn'][:,0])
nino34_years = np.arange(1873,2008)



### CALCULATIONS

# For LMR and CCSM4 prior pdsi, set values of 0 to nan
scpdsi_lmr_all[scpdsi_lmr_all == 0] = np.nan
scpdsi_ccsm4_mean[scpdsi_ccsm4_mean == 0] = np.nan

# Compute the mean of all LMR iterations.
scpdsi_lmr_mean = np.mean(scpdsi_lmr_all,axis=1)
nino34_lmr_mean = np.mean(nino34_lmr_all,axis=1)

# Mask nans
scpdsi_lmr_mean = ma.masked_invalid(scpdsi_lmr_mean)
scpdsi_ccsm4_mean = ma.masked_invalid(scpdsi_ccsm4_mean)


# Compute annual mean values for the DaiPDSI data set
# DaiPDSI data runs from January 1850 to December 2014
nmonths_dai = pdsi_dai_monthly.shape[0]
nlat_dai    = lat_dai.shape[0]
nlon_dai    = lon_dai.shape[0]
pdsi_dai_monthly_2d = np.reshape(pdsi_dai_monthly,(int(nmonths_dai/12),12,nlat_dai,nlon_dai))

pdsi_dai_annual = np.zeros((years_dai.shape[0],nlat_dai,nlon_dai)); pdsi_dai_annual[:] = np.nan
for j in range(nlat_dai):
    print('Computing annual means: '+str(j+1)+'/'+str(nlat_dai))
    for i in range(nlon_dai):
        pdsi_dai_annual[:,j,i] = mpe.annual_mean(years_dai,pdsi_dai_monthly_2d[:,:,j,i])
        
# Reorganize the DaiPDSI dataset so that longitude runs from 0 to 360 instead of -180 to 180.
indices_east = np.where(lon_dai > 0)[0]
indices_west = np.where(lon_dai < 0)[0]

lon_dai_east = lon_dai[indices_east]
lon_dai_west = lon_dai[indices_west]+360
lon_dai = np.concatenate((lon_dai_east,lon_dai_west),axis=0)

pdsi_dai_east = pdsi_dai_annual[:,:,indices_east]
pdsi_dai_west = pdsi_dai_annual[:,:,indices_west]
pdsi_dai_annual = np.concatenate((pdsi_dai_east,pdsi_dai_west),axis=2)


# Compute Nino3.4 averages over several time periods
# Bunge & Clark data runs from January 1873 to March 2008

# Select only the 1873-2007 data
nino34_monthly = nino34_monthly[0:1620]
nino34_ndays   = nino34_ndays[0:1620]
nyears = len(nino34_years)

# Annual mean
nino34_2d       = np.reshape(nino34_monthly,(nyears,12))
nino34_ndays_2d = np.reshape(nino34_ndays,  (nyears,12))
nino34_annual = np.average(nino34_2d,axis=1,weights=nino34_ndays_2d)

# MAM mean
nino34_mam = np.average(nino34_2d[:,2:5],axis=1,weights=nino34_ndays_2d[:,2:5])

# DJF mean
nino34_monthly_padded = np.insert(nino34_monthly,0,np.nan)  # I'm missing a December, so insert a nan to help with the reshaping.
nino34_ndays_padded   = np.insert(nino34_ndays,0,31)
nino34_2d_padded       = np.reshape(nino34_monthly_padded[0:1620],(nyears,12))
nino34_ndays_2d_padded = np.reshape(nino34_ndays_padded[0:1620],  (nyears,12))
nino34_djf = np.average(nino34_2d_padded[:,0:3],axis=1,weights=nino34_ndays_2d_padded[:,0:3])



# More data managment
# Mask the PDSI
pdsi_dai_annual = ma.masked_invalid(pdsi_dai_annual)

# Shorten both observational datasets to 1874-2000 (exclude 1873 because the DJF from that years is nan)
year_start = 1874
year_end   = 2000
years_obs = np.arange(year_start,year_end+1)

index1_dai = np.where(years_dai == year_start)[0][0]
index2_dai = np.where(years_dai == year_end)[0][0]
years_dai_selected       = years_dai[index1_dai:index2_dai+1]
pdsi_dai_annual_selected = pdsi_dai_annual[index1_dai:index2_dai+1,:,:]

index1_nino34 = np.where(nino34_years == year_start)[0][0]
index2_nino34 = np.where(nino34_years == year_end)[0][0]
nino34_years_selected        = nino34_years[index1_nino34:index2_nino34+1]
nino34_annual_selected       = nino34_annual[index1_nino34:index2_nino34+1]
nino34_mam_selected          = nino34_mam[index1_nino34:index2_nino34+1]
nino34_djf_selected          = nino34_djf[index1_nino34:index2_nino34+1]
nino34_annual_lead1_selected = nino34_annual[index1_nino34-1:index2_nino34]


# Set LMR variables to the right lengths.
#year_start_lmr = 1001
year_start_lmr = 1874
year_end_lmr   = 2000
years_lmr_selected = np.arange(year_start_lmr,year_end_lmr+1)

index1_lmr = np.where(years_lmr == year_start_lmr)[0][0]
index2_lmr = np.where(years_lmr == year_end_lmr)[0][0]

scpdsi_lmr_selected       = scpdsi_lmr_mean[index1_lmr:index2_lmr+1,:,:]
nino34_lmr_selected       = nino34_lmr_mean[index1_lmr:index2_lmr+1]
nino34_lmr_lead1_selected = nino34_lmr_mean[index1_lmr-1:index2_lmr]


# Since the CCSM4 data doesn't have the same years, select the same number of years from the end of the simulation.
nyears_lmr = len(nino34_lmr_selected)
scpdsi_ccsm4_selected       = scpdsi_ccsm4_mean[-1*nyears_lmr:,:,:]
nino34_ccsm4_selected       = nino34_ccsm4_mean[-1*nyears_lmr:]
nino34_ccsm4_lead1_selected = nino34_ccsm4_mean[-1*nyears_lmr-1:-1]


# Compute correlations between Nino3.4 averaged over different time periods and PDSI at every location.
nlat_dai   = len(lat_dai)
nlon_dai   = len(lon_dai)
nlat_lmr   = len(lat_lmr)
nlon_lmr   = len(lon_lmr)
nlat_ccsm4 = len(lat_ccsm4)
nlon_ccsm4 = len(lon_ccsm4)
correlations = {}
correlations['obs_annual']         = np.zeros((nlat_dai,nlon_dai));     correlations['obs_annual'][:]         = np.nan
correlations['obs_annual_lead1']   = np.zeros((nlat_dai,nlon_dai));     correlations['obs_annual_lead1'][:]   = np.nan
correlations['obs_mam']            = np.zeros((nlat_dai,nlon_dai));     correlations['obs_mam'][:]            = np.nan
correlations['obs_djf']            = np.zeros((nlat_dai,nlon_dai));     correlations['obs_djf'][:]            = np.nan
correlations['lmr_annual']         = np.zeros((nlat_lmr,nlon_lmr));     correlations['lmr_annual'][:]         = np.nan
correlations['lmr_annual_lead1']   = np.zeros((nlat_lmr,nlon_lmr));     correlations['lmr_annual_lead1'][:]   = np.nan
correlations['ccsm4_annual']       = np.zeros((nlat_ccsm4,nlon_ccsm4)); correlations['ccsm4_annual'][:]       = np.nan
correlations['ccsm4_annual_lead1'] = np.zeros((nlat_ccsm4,nlon_ccsm4)); correlations['ccsm4_annual_lead1'][:] = np.nan

# Loop through every point, calculating correlations.
for j in range(nlat_dai):
    print('Calculating coefficients for obs.: '+str(j+1)+'/'+str(nlat_dai))
    for i in range(nlon_dai):
        correlations['obs_annual'][j,i]       = np.ma.corrcoef(nino34_annual_selected,      pdsi_dai_annual_selected[:,j,i])[0,1]
        correlations['obs_annual_lead1'][j,i] = np.ma.corrcoef(nino34_annual_lead1_selected,pdsi_dai_annual_selected[:,j,i])[0,1]
        correlations['obs_mam'][j,i]          = np.ma.corrcoef(nino34_mam_selected,         pdsi_dai_annual_selected[:,j,i])[0,1]
        correlations['obs_djf'][j,i]          = np.ma.corrcoef(nino34_djf_selected,         pdsi_dai_annual_selected[:,j,i])[0,1]

for j in range(nlat_lmr):
    print('Calculating coefficients for LMR: '+str(j+1)+'/'+str(nlat_lmr))
    for i in range(nlon_lmr):
        correlations['lmr_annual'][j,i]       = np.ma.corrcoef(nino34_lmr_selected,      scpdsi_lmr_selected[:,j,i])[0,1]
        correlations['lmr_annual_lead1'][j,i] = np.ma.corrcoef(nino34_lmr_lead1_selected,scpdsi_lmr_selected[:,j,i])[0,1]

for j in range(nlat_ccsm4):
    print('Calculating coefficients for CCSM4 prior: '+str(j+1)+'/'+str(nlat_ccsm4))
    for i in range(nlon_ccsm4):
        correlations['ccsm4_annual'][j,i]       = np.ma.corrcoef(nino34_ccsm4_selected,      scpdsi_ccsm4_selected[:,j,i])[0,1]
        correlations['ccsm4_annual_lead1'][j,i] = np.ma.corrcoef(nino34_ccsm4_lead1_selected,scpdsi_ccsm4_selected[:,j,i])[0,1]


# Average over specific regions
pdsi_dai_mean   = {}
pdsi_lmr_mean   = {}
pdsi_ccsm4_mean = {}
for region in possible_regions:
    #
    print('Averaging over '+region)
    lat_min = region_bounds[region][0]
    lat_max = region_bounds[region][1]
    lon_min = region_bounds[region][2]
    lon_max = region_bounds[region][3]
    #
    pdsi_dai_mean[region]   = compute_regional_means.compute_means(pdsi_dai_annual_selected,lat_dai,  lon_dai,  lat_min,lat_max,lon_min,lon_max)
    pdsi_lmr_mean[region]   = compute_regional_means.compute_means(scpdsi_lmr_selected,     lat_lmr,  lon_lmr,  lat_min,lat_max,lon_min,lon_max)
    pdsi_ccsm4_mean[region] = compute_regional_means.compute_means(scpdsi_ccsm4_selected,   lat_ccsm4,lon_ccsm4,lat_min,lat_max,lon_min,lon_max)

# Calculating correlations (and R^2 values) for specific regions.
r_regions_obs_annual         = {}
r_regions_obs_annual_lead1   = {}
r_regions_obs_mam            = {}
r_regions_obs_djf            = {}
r_regions_lmr_annual         = {}
r_regions_lmr_annual_lead1   = {}
r_regions_ccsm4_annual       = {}
r_regions_ccsm4_annual_lead1 = {}
for region in possible_regions:
    r_regions_obs_annual[region]         = np.ma.corrcoef(nino34_annual_selected,      pdsi_dai_mean[region])[0,1]
    r_regions_obs_annual_lead1[region]   = np.ma.corrcoef(nino34_annual_lead1_selected,pdsi_dai_mean[region])[0,1]
    r_regions_obs_mam[region]            = np.ma.corrcoef(nino34_mam_selected,         pdsi_dai_mean[region])[0,1]
    r_regions_obs_djf[region]            = np.ma.corrcoef(nino34_djf_selected,         pdsi_dai_mean[region])[0,1]
    r_regions_lmr_annual[region]         = np.ma.corrcoef(nino34_lmr_selected,         pdsi_lmr_mean[region])[0,1]
    r_regions_lmr_annual_lead1[region]   = np.ma.corrcoef(nino34_lmr_lead1_selected,   pdsi_lmr_mean[region])[0,1]
    r_regions_ccsm4_annual[region]       = np.ma.corrcoef(nino34_ccsm4_selected,       pdsi_ccsm4_mean[region])[0,1]
    r_regions_ccsm4_annual_lead1[region] = np.ma.corrcoef(nino34_ccsm4_lead1_selected, pdsi_ccsm4_mean[region])[0,1]

format = '%30s %1.2f %2s %1.2f'
print(' === Regions: '+possible_regions[0]+' and '+possible_regions[1]+' ===')
print(format % ('Obs, annual, R^2: ',        np.square(r_regions_obs_annual[possible_regions[0]]),        ', ',np.square(r_regions_obs_annual[possible_regions[1]])))
print(format % ('Obs, annual lead1, R^2: ',  np.square(r_regions_obs_annual_lead1[possible_regions[0]]),  ', ',np.square(r_regions_obs_annual_lead1[possible_regions[1]])))
print(format % ('Obs, mam, R^2: ',           np.square(r_regions_obs_mam[possible_regions[0]]),           ', ',np.square(r_regions_obs_mam[possible_regions[1]])))
print(format % ('Obs, djf, R^2: ',           np.square(r_regions_obs_djf[possible_regions[0]]),           ', ',np.square(r_regions_obs_djf[possible_regions[1]])))
print(format % ('LMR, annual, R^2: ',        np.square(r_regions_lmr_annual[possible_regions[0]]),        ', ',np.square(r_regions_lmr_annual[possible_regions[1]])))
print(format % ('LMR, annual lead1, R^2: ',  np.square(r_regions_lmr_annual_lead1[possible_regions[0]]),  ', ',np.square(r_regions_lmr_annual_lead1[possible_regions[1]])))
print(format % ('CCSM4, annual, R^2: ',      np.square(r_regions_ccsm4_annual[possible_regions[0]]),      ', ',np.square(r_regions_ccsm4_annual[possible_regions[1]])))
print(format % ('CCSM4, annual lead1, R^2: ',np.square(r_regions_ccsm4_annual_lead1[possible_regions[0]]),', ',np.square(r_regions_ccsm4_annual_lead1[possible_regions[1]])))


# Regrid correlation maps to the same grid
lon_lmr_2d, lat_lmr_2d = np.meshgrid(lon_lmr, lat_lmr)

# Regrid the oservational output to the grid of the lmr
correlations['obs_annual_regrid']       = basemap.interp(correlations['obs_annual'],       lon_dai, lat_dai, lon_lmr_2d, lat_lmr_2d, order=1)
correlations['obs_annual_lead1_regrid'] = basemap.interp(correlations['obs_annual_lead1'], lon_dai, lat_dai, lon_lmr_2d, lat_lmr_2d, order=1)

# Regrid the ccsm4 output to the grid of the lmr
correlations['ccsm4_annual_regrid']       = basemap.interp(correlations['ccsm4_annual'],       lon_ccsm4, lat_ccsm4, lon_lmr_2d, lat_lmr_2d, order=1)
correlations['ccsm4_annual_lead1_regrid'] = basemap.interp(correlations['ccsm4_annual_lead1'], lon_ccsm4, lat_ccsm4, lon_lmr_2d, lat_lmr_2d, order=1)

# For the DaiPDSI dataset, for lats poleward of 78N, set to nan, since the DaiPDSI dataset doesn't have data that far north.
index_78N = np.where(lat_lmr==78)[0][0]
correlations['obs_annual_regrid'][index_78N:,:]       = np.nan
correlations['obs_annual_lead1_regrid'][index_78N:,:] = np.nan

# Calculate differences and RMSE between LMR and the prior and LMR and the obs.
correlations['diff_lmr_obs']         = correlations['lmr_annual']       - correlations['obs_annual_regrid']
correlations['diff_lmr_obs_lead1']   = correlations['lmr_annual_lead1'] - correlations['obs_annual_lead1_regrid']
correlations['diff_lmr_ccsm4']       = correlations['lmr_annual']       - correlations['ccsm4_annual_regrid']
correlations['diff_lmr_ccsm4_lead1'] = correlations['lmr_annual_lead1'] - correlations['ccsm4_annual_lead1_regrid']

# Mask our regions which are nan in any of the three data sets.
sum_annual       = correlations['lmr_annual']       + correlations['obs_annual_regrid']       + correlations['ccsm4_annual_regrid']
sun_annual_lead1 = correlations['lmr_annual_lead1'] + correlations['obs_annual_lead1_regrid'] + correlations['ccsm4_annual_lead1_regrid']
correlations['diff_lmr_obs'][np.isnan(sum_annual)]               = np.nan
correlations['diff_lmr_obs_lead1'][np.isnan(sun_annual_lead1)]   = np.nan
correlations['diff_lmr_ccsm4'][np.isnan(sum_annual)]             = np.nan
correlations['diff_lmr_ccsm4_lead1'][np.isnan(sun_annual_lead1)] = np.nan

# Calculate mean differences over certain regions.
diff_region1 = [15,65,220,305]
diff_region2 = [30,49,230,295]  # 30-49N, 130-65W
diff_region3 = [28,40,245,265]  # 28-40N, 115-95W
diff_keys = ['diff_lmr_obs','diff_lmr_obs_lead1','diff_lmr_ccsm4','diff_lmr_ccsm4_lead1']

lat_min1 = diff_region1[0]; lat_max1 = diff_region1[1]; lon_min1 = diff_region1[2]; lon_max1 = diff_region1[3]
lat_min2 = diff_region2[0]; lat_max2 = diff_region2[1]; lon_min2 = diff_region2[2]; lon_max2 = diff_region2[3]
lat_min3 = diff_region3[0]; lat_max3 = diff_region3[1]; lon_min3 = diff_region3[2]; lon_max3 = diff_region3[3]

mean_diff_region1 = {}; RMSE_region1 = {}
mean_diff_region2 = {}; RMSE_region2 = {}
mean_diff_region3 = {}; RMSE_region3 = {}
for key in diff_keys:
    print('Averaging for '+key)
    mean_diff_region1[key] =         compute_regional_means.compute_means(np.abs(correlations[key][None,:,:]),   lat_lmr,lon_lmr,lat_min1,lat_max1,lon_min1,lon_max1)
    RMSE_region1[key]      = np.sqrt(compute_regional_means.compute_means(np.square(correlations[key][None,:,:]),lat_lmr,lon_lmr,lat_min1,lat_max1,lon_min1,lon_max1))
    mean_diff_region2[key] =         compute_regional_means.compute_means(np.abs(correlations[key][None,:,:]),   lat_lmr,lon_lmr,lat_min2,lat_max2,lon_min2,lon_max2)
    RMSE_region2[key]      = np.sqrt(compute_regional_means.compute_means(np.square(correlations[key][None,:,:]),lat_lmr,lon_lmr,lat_min2,lat_max2,lon_min2,lon_max2))
    mean_diff_region3[key] =         compute_regional_means.compute_means(np.abs(correlations[key][None,:,:]),   lat_lmr,lon_lmr,lat_min3,lat_max3,lon_min3,lon_max3)
    RMSE_region3[key]      = np.sqrt(compute_regional_means.compute_means(np.square(correlations[key][None,:,:]),lat_lmr,lon_lmr,lat_min3,lat_max3,lon_min3,lon_max3))



timemeans_diff = ['diff_lmr_obs','diff_lmr_obs_lead1','diff_lmr_ccsm4','diff_lmr_ccsm4_lead1']
titles_diff    = ['LMR vs. Obs.','LMR vs. Obs., Nino3.4 leading by 1 year','LMR vs. CCSM4','LMR vs. CCSM4, Nino3.4 leading by 1 year']

limit = 1
levels = np.linspace(-1*limit,limit,21)

# FIGURE: Plot a map of the difference in correlations.
f, ax = plt.subplots(2,2,figsize=(16,16))
ax = ax.ravel()

for i,timemean in enumerate(timemeans_diff):
    #
    ax[i].set_title(titles_diff[i],fontsize=20)
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=65,llcrnrlon=220,urcrnrlon=305,resolution='c',ax=ax[i])
    lon_lmr_2d,lat_lmr_2d = np.meshgrid(lon_lmr,lat_lmr)
    x_lmr,y_lmr = m(lon_lmr_2d,lat_lmr_2d)
    map1 = m.contourf(x_lmr,y_lmr,np.ma.masked_array(correlations[timemean],np.isnan(correlations[timemean])),levels,cmap='RdBu_r',vmin=-1*limit,vmax=limit)
    m.colorbar(map1,ax=ax[i],location='bottom').ax.tick_params(labelsize=16)
    m.drawcoastlines()
    #
    # Draw a box for the region of interest
    for region in [diff_region2,diff_region3]:
        box_lat_min, box_lat_max, box_lon_min, box_lon_max = region
        x_region,y_region = m([box_lon_min,box_lon_min,box_lon_max,box_lon_max],[box_lat_min,box_lat_max,box_lat_max,box_lat_min])
        xy_region = np.column_stack((x_region,y_region))
        region_box = Polygon(xy_region,edgecolor='black',facecolor='none',linewidth=2,alpha=.5)
        plt.gca().add_patch(region_box)
    #
    # List the mean diff and RMSE values for the region.
    plt.text(.02,.08,'Entire region            : mean_diff = '+str('%1.2f' % mean_diff_region1[timemean])+', RMSE = '+str('%1.2f' % RMSE_region1[timemean]),transform=ax[i].transAxes,fontsize=12)
    plt.text(.02,.05,'US region                 : mean_diff = '+str('%1.2f' % mean_diff_region2[timemean])+', RMSE = '+str('%1.2f' % RMSE_region2[timemean]),transform=ax[i].transAxes,fontsize=12)
    plt.text(.02,.02,'Southwest US region: mean_diff = '+str('%1.2f' % mean_diff_region3[timemean])+', RMSE = '+str('%1.2f' % RMSE_region3[timemean]),transform=ax[i].transAxes,fontsize=12)

f.suptitle("Difference in correlations maps",fontsize=20)
f.tight_layout()
f.subplots_adjust(top=.93)
if save_instead_of_plot:
    plt.savefig("figures/correlation_maps_differences_years_"+str(year_start_lmr)+"_"+str(year_end_lmr)+".png",dpi=300,format='png')
else:
    plt.show()






### FIGURES
plt.style.use('ggplot')

timemeans = ['obs_annual','obs_annual_lead1','obs_djf',\
             'ccsm4_annual','ccsm4_annual_lead1','',\
             'lmr_annual','lmr_annual_lead1']
titles    = ['a) Obs., annual Nino3.4','b) Obs., annual Nino3.4 leading by 1yr','c) Obs., DJF Nino3.4',\
             'd) CCSM4, annual Nino3.4','e) CCSM4, annual Nino3.4 leading by 1yr','',\
             'f) LMR, annual Nino3.4','g) LMR, annual Nino3.4 leading by 1yr']
chosen_region = 'Southwest US'
r_values  = [r_regions_obs_annual[chosen_region],r_regions_obs_annual_lead1[chosen_region],r_regions_obs_djf[chosen_region],\
             r_regions_ccsm4_annual[chosen_region],r_regions_ccsm4_annual_lead1[chosen_region],'',\
             r_regions_lmr_annual[chosen_region],r_regions_lmr_annual_lead1[chosen_region]]


limit = 1
levels = np.linspace(-1*limit,limit,21)


# FIGURE: Plot a map of the correlation between nino3.4 at differnet time averages and scpdsi.
f, ax = plt.subplots(3,3,figsize=(24,24))
ax = ax.ravel()

for i,timemean in enumerate(timemeans):
    #
    ax[i].set_title(titles[i],fontsize=20,loc='left')
    #
    box_lat_min, box_lat_max, box_lon_min, box_lon_max = region_bounds['Southwest US']
    #
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=65,llcrnrlon=220,urcrnrlon=305,resolution='c',ax=ax[i])
    if i < 3:
        lon_dai_2d,lat_dai_2d = np.meshgrid(lon_dai,lat_dai)
        x_dai,y_dai = m(lon_dai_2d,lat_dai_2d)
        map1 = m.contourf(x_dai,y_dai,np.ma.masked_array(correlations[timemean],np.isnan(correlations[timemean])),levels,cmap='RdBu_r',vmin=-1*limit,vmax=limit)
    elif i < 5:
        lon_ccsm4_2d,lat_ccsm4_2d = np.meshgrid(lon_ccsm4,lat_ccsm4)
        x_ccsm4,y_ccsm4 = m(lon_ccsm4_2d,lat_ccsm4_2d)
        map1 = m.contourf(x_ccsm4,y_ccsm4,np.ma.masked_array(correlations[timemean],np.isnan(correlations[timemean])),levels,cmap='RdBu_r',vmin=-1*limit,vmax=limit)
    elif i == 5:
        continue
    else:
        lon_lmr_2d,lat_lmr_2d = np.meshgrid(lon_lmr,lat_lmr)
        x_lmr,y_lmr = m(lon_lmr_2d,lat_lmr_2d)
        map1 = m.contourf(x_lmr,y_lmr,np.ma.masked_array(correlations[timemean],np.isnan(correlations[timemean])),levels,cmap='RdBu_r',vmin=-1*limit,vmax=limit)
    #
    box_lon_min = box_lon_min+360
    box_lon_max = box_lon_max+360
    m.colorbar(map1,ax=ax[i],location='bottom').ax.tick_params(labelsize=16)
    m.drawcoastlines()
    #
    # Draw a box for the region of interest
    x_region,y_region = m([box_lon_min,box_lon_min,box_lon_max,box_lon_max],[box_lat_min,box_lat_max,box_lat_max,box_lat_min])
    xy_region = np.column_stack((x_region,y_region))
    region_box = Polygon(xy_region,edgecolor='black',facecolor='none',linewidth=2,alpha=.5)
    plt.gca().add_patch(region_box)
    #
    # List the R-squared value for the region
    plt.text(.02,.03,'$R_{region}$ = '+str('%1.2f' % r_values[i]),transform=ax[i].transAxes,fontsize=22)

# Two panels aren't being used, so hide them.
ax[5].axis('off')
ax[8].axis('off')

f.suptitle("Correlations between Nino3.4 at different time means and PDSI, using observations, CCSM4, and LMR",fontsize=28)
f.tight_layout()
f.subplots_adjust(top=.95)
if save_instead_of_plot:
    plt.savefig("figures/correlation_maps_obs_ccsm4_lmr_years_"+str(year_start_lmr)+"_"+str(year_end_lmr)+".png",dpi=300,format='png')
else:
    plt.show()

