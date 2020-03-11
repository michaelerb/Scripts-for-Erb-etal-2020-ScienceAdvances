#=============================================================================
# This script looks at the climate patterns for the driest and wettest PDSI
# events in the south-west U.S.
#
# To do:
#    - Figure out why the 500hPa height anomlies look strange (i.e. why does
#      the N. Pacific high pass through 0, rather than being above 0 the whole time?).
#
#    author: Michael P. Erb
#    date  : 7/5/2019
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions/field_correlation')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import xarray as xr
from scipy.stats.stats import pearsonr
from matplotlib.patches import Polygon
import lmr_land_ocean_masks as masks
import compute_regional_means
from corr_2d_ttest import *
from corr_sig import *
import time


save_instead_of_plot = True

remove_influence_of_nino34 = False
remove_influence_of_pdo    = False

# Choose the type of correlation calculation to preform
#corr_calc_type = 'pearsonr'
corr_calc_type = 'isospectral'

# Choose the variable of interest: precip, pdsi
reference_variable = 'PDSI'
#reference_variable = 'precipitation'

# If desried, use only the years with/without data at Palmyra.
year_filter = "all years"
#year_filter = "only palmyra"
#year_filter = "no palmyra"

# Specify the region to compute the calculation over
calc_bounds = [-25,70,90,360]

# Specify the desired regions.  Specify bounds at lat_min, lat_max, lon_min, lon_max
#regions = ['US','Northwest US','Southwest US','California','Central US','Southeast US']
regions = ['Northwest US','Southwest US','Central US','Southeast US']
region_bounds = {}
region_bounds['US']               = [30,  49,-130,   -65]  # 30-49N, 130-65W
region_bounds['South-central US'] = [28,  40,-115,   -95]  # 28-40N, 115-95W
region_bounds['Northwest US']     = [42,  50,-125,  -110]  # 42-50N, 125-110W
region_bounds['Southwest US']     = [32,  40,-125,  -105]  # 32-40N, 125-105W
region_bounds['California']       = [32.5,42,-124.5,-117]  # 32.5-42N, 124.5-117W
region_bounds['Central US']       = [34,  46,-102,   -92]  # 34-46N, 102- 92W
region_bounds['Southeast US']     = [30,  39, -92,   -75]  # 30-39N,  92- 75W


### FUNCTIONS

# Compute annual-means
#var,days_per_month = scpdsi_all_prior,days_per_month_prior
def annual_mean(var,days_per_month):
    ntime = var.shape[0]
    nlat  = var.shape[1]
    nlon  = var.shape[2]
    nyears = np.int(ntime/12)
    var_2d = np.reshape(var,(nyears,12,nlat,nlon))
    days_per_month_2d = np.reshape(days_per_month,(nyears,12))
    #
    var_annual = np.zeros((nyears,nlat,nlon)); var_annual[:] = np.nan
    for i in range(nyears):
        var_annual[i,:,:] = np.average(var_2d[i,:,:,:],axis=0,weights=days_per_month_2d[i,:])
    #
    return var_annual



### LOAD DATA

# Load data from the production run
data_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name1_prod = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
experiment_name2_prod = experiment_name1_prod

handle = xr.open_dataset(data_dir+experiment_name1_prod+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_all_prod = handle['pdsi'].values
lon_prod        = handle['lon'].values
lat_prod        = handle['lat'].values
time_prod       = handle['time'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2_prod+'/sst_MCruns_ensemble_mean.nc',decode_times=False)
sst_all_prod = handle['sst'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2_prod+'/hgt500_MCruns_ensemble_mean.nc',decode_times=False)
zg_500hPa_all_prod = handle['hgt500'].values
handle.close()

years_prod = time_prod/365
years_prod = years_prod.astype(int)

landmask_prod, oceanmask_prod = masks.masks(data_dir+experiment_name1_prod+'/r0/ensemble_mean_tas_sfc_Amon.npz')



# Load data from the NAonly/NAexclude run
experiment_name1_na = 'LMR_gisgpcc_ccms4_LMRdbv0.4.0_NorthAmerica_only'
experiment_name2_na = 'LMR_gisgpcc_ccms4_LMRdbv0.4.0_NorthAmerica_exclude'

handle = xr.open_dataset(data_dir+experiment_name1_na+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_all_na = handle['pdsi'].values
lon_na        = handle['lon'].values
lat_na        = handle['lat'].values
time_na       = handle['time'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2_na+'/sst_MCruns_ensemble_mean.nc',decode_times=False)
sst_all_na = handle['sst'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2_na+'/hgt500_MCruns_ensemble_mean.nc',decode_times=False)
zg_500hPa_all_na = handle['hgt500'].values
handle.close()

years_na = time_na/365
years_na = years_na.astype(int)

landmask_na, oceanmask_na = masks.masks(data_dir+experiment_name1_na+'/r0/ensemble_mean_tas_sfc_Amon.npz')



# Load data from the prior
data_dir_prior        = '/projects/pd_lab/data/LMR/data/model/ccsm4_last_millenium/'
data_dir_regrid_prior = '/projects/pd_lab/data/processed_data/LMR_regrid/data_regrid/'

handle = xr.open_dataset(data_dir_prior+'scpdsipm_sfc_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
scpdsi_all_prior = handle['scpdsipm'].values
lon_prior        = handle['lon'].values
lat_prior        = handle['lat'].values
time_bnds_prior  = handle['time_bnds'].values
handle.close()

handle = xr.open_dataset(data_dir_regrid_prior+'tos_sfc_Omon_CCSM4_past1000_085001-185012_regrid.nc',decode_times=False)
sst_all_prior = handle['tos'].values
handle.close()

handle = xr.open_dataset(data_dir_prior+'zg_500hPa_Amon_CCSM4_past1000_085001-185012.nc',decode_times=False)
zg_500hPa_all_prior = np.squeeze(handle['zg'].values)
handle.close()

# Compute annual-means
days_per_month_prior = time_bnds_prior[:,1]-time_bnds_prior[:,0]
scpdsi_mean_prior    = annual_mean(scpdsi_all_prior,   days_per_month_prior)
sst_mean_prior       = annual_mean(sst_all_prior,      days_per_month_prior)
zg_500hPa_mean_prior = annual_mean(zg_500hPa_all_prior,days_per_month_prior)

years_prior = np.arange(850,1851)



### CALCULATIONS

# Remove the mean value from the heights.
zg_500hPa_all_mean_prod = np.mean(zg_500hPa_all_prod,axis=0)
zg_500hPa_all_prod = zg_500hPa_all_prod-zg_500hPa_all_mean_prod[None,:,:,:]
zg_500hPa_all_mean_na = np.mean(zg_500hPa_all_na,axis=0)
zg_500hPa_all_na = zg_500hPa_all_na-zg_500hPa_all_mean_na[None,:,:,:]

# Compute a mean of all iterations
scpdsi_mean_prod    = np.mean(scpdsi_all_prod,axis=1)
sst_mean_prod       = np.mean(sst_all_prod,axis=1)
zg_500hPa_mean_prod = np.mean(zg_500hPa_all_prod,axis=1)
scpdsi_mean_na    = np.mean(scpdsi_all_na,axis=1)
sst_mean_na       = np.mean(sst_all_na,axis=1)
zg_500hPa_mean_na = np.mean(zg_500hPa_all_na,axis=1)

# Mask the variables
scpdsi_mean_prod = scpdsi_mean_prod*oceanmask_prod[None,:,:]
scpdsi_mean_prod = ma.masked_invalid(scpdsi_mean_prod)
scpdsi_mean_na = scpdsi_mean_na*oceanmask_na[None,:,:]
scpdsi_mean_na = ma.masked_invalid(scpdsi_mean_na)

# Function to reduce the climate fields to the desired region
#var_chosen,lat_chosen,lon_chosen = scpdsi_mean_prod,lat_prod,lon_prod
def select_region(var_chosen,lat_chosen,lon_chosen):
    #
    global calc_bounds
    #
    j_min = np.abs(lat_chosen-calc_bounds[0]).argmin()
    j_max = np.abs(lat_chosen-calc_bounds[1]).argmin()
    i_min = np.abs(lon_chosen-calc_bounds[2]).argmin()
    i_max = np.abs(lon_chosen-calc_bounds[3]).argmin()
    print('Indices chosen. lat='+str(j_min)+'-'+str(j_max)+', lon='+str(i_min)+'-'+str(i_max))
    #
    var_selected = var_chosen[:,j_min:j_max+1,i_min:i_max+1]
    lat_selected = lat_chosen[j_min:j_max+1]
    lon_selected = lon_chosen[i_min:i_max+1]
    #
    return var_selected,lat_selected,lon_selected

# Reduce the climate fields to the desired region
scpdsi_mean_prod_selected,   lat_prod_selected,lon_prod_selected = select_region(scpdsi_mean_prod,   lat_prod,lon_prod)
sst_mean_prod_selected,      lat_prod_selected,lon_prod_selected = select_region(sst_mean_prod,      lat_prod,lon_prod)
zg_500hPa_mean_prod_selected,lat_prod_selected,lon_prod_selected = select_region(zg_500hPa_mean_prod,lat_prod,lon_prod)

scpdsi_mean_na_selected,   lat_na_selected,lon_na_selected = select_region(scpdsi_mean_na,   lat_na,lon_na)
sst_mean_na_selected,      lat_na_selected,lon_na_selected = select_region(sst_mean_na,      lat_na,lon_na)
zg_500hPa_mean_na_selected,lat_na_selected,lon_na_selected = select_region(zg_500hPa_mean_na,lat_na,lon_na)

scpdsi_mean_prior_selected,   lat_prior_selected,lon_prior_selected = select_region(scpdsi_mean_prior,   lat_prior,lon_prior)
sst_mean_prior_selected,      lat_prior_selected,lon_prior_selected = select_region(sst_mean_prior,      lat_prior,lon_prior)
zg_500hPa_mean_prior_selected,lat_prior_selected,lon_prior_selected = select_region(zg_500hPa_mean_prior,lat_prior,lon_prior)


# For every region, compute the correlation between drought in that region and SSTs and 500hPa heights.
#var_ts,var_spatial = var_regional_mean,sst_mean
def correlation_calc(var_ts,var_spatial,lat,lon):
    #
    global regions
    #
    nlat = var_spatial.shape[1]
    nlon = var_spatial.shape[2]
    #
    correlations = {}
    pvalues      = {}
    lat_not_sig  = {}
    lon_not_sig  = {}
    for region in regions:
        print('Calculating correlations for region: '+region)
        correlations[region] = np.zeros((nlat,nlon)); correlations[region][:] = np.nan
        pvalues[region]      = np.zeros((nlat,nlon)); pvalues[region][:]      = np.nan
        for j in range(nlat):
            for i in range(nlon):
                correlations[region][j,i],pvalues[region][j,i] = pearsonr(var_ts[region],var_spatial[:,j,i])
        #
        # Create a list of lats and lons for values above a certain threshold
        lat_not_sig[region] = []
        lon_not_sig[region] = []
        for j,lat_chosen in enumerate(lat):
            for i,lon_chosen in enumerate(lon):
                if pvalues[region][j,i] >= 0.05:
                    lat_not_sig[region].append(lat_chosen)
                    lon_not_sig[region].append(lon_chosen)
    #
    return correlations,pvalues,lat_not_sig,lon_not_sig


# Function to remove points over land
def remove_points_over_land(sst_mean,lat_not_sig_sst,lon_not_sig_sst,lat,lon):
    #
    global regions
    #
    lat_not_sig_noland = {}
    lon_not_sig_noland = {}
    #
    for region in regions:
        lats_candidate = lat_not_sig_sst[region]
        lons_candidate = lon_not_sig_sst[region]
        lat_not_sig_noland[region] = []
        lon_not_sig_noland[region] = []
        #
        npoints = len(lats_candidate)
        for index in range(npoints):
            j_selected = np.argwhere(lat == lats_candidate[index])
            i_selected = np.argwhere(lon == lons_candidate[index])
            if not np.isnan(np.mean(sst_mean[:,j_selected,i_selected])):
                lat_not_sig_noland[region].append(lats_candidate[index])
                lon_not_sig_noland[region].append(lons_candidate[index])
    #
    return lat_not_sig_noland,lon_not_sig_noland


# Make a function that calculates correlations, then makes a figure as well as outputing the variables
#dataset,scpdsi_mean,sst_mean,zg_500hPa_mean,lat,lon,years,year_bounds = 'LMR posterior',scpdsi_mean_prod_selected,sst_mean_prod_selected,zg_500hPa_mean_prod_selected,lat_prod_selected,lon_prod_selected,years_prod,[1001,2000]
#dataset,scpdsi_mean,sst_mean,zg_500hPa_mean,lat,lon,years,year_bounds = 'CCSM4 prior',  scpdsi_mean_prior_selected,sst_mean_prior_selected,zg_500hPa_mean_prior_selected,lat_prior_selected,lon_prior_selected,years_prior,[850,1850]
def correlation_maps(dataset,scpdsi_mean,sst_mean,zg_500hPa_mean,lat,lon,years,year_bounds):
    #
    # Import global variables
    global regions,region_bounds,calc_bounds,remove_influence_of_nino34,remove_influence_of_pdo,year_filter,save_instead_of_plot
    #
    # Set the variable of interest
    # The variables ares inverted so that the correlations show what's correlated to dry conditions, not wet conditions.
    reference_variable = 'PDSI'
    var_mean = -1*scpdsi_mean
    #
    # Load the Palmyra data and determine which years it covers.
    palmyra_data = np.loadtxt('/home/mpe32/analysis/5_drought/more/data/data_palmyra.txt')
    palmyra_years_with_data = palmyra_data[~np.isnan(palmyra_data[:,1]),0]
    palmyra_years_with_data_inrange = palmyra_years_with_data[(palmyra_years_with_data>=years[0]) & (palmyra_years_with_data<=years[-1])]
    #
    # Shorten the data to cover only the desired years.
    indices_chosen = np.where((years >= year_bounds[0]) & (years <= year_bounds[1]))[0]
    var_mean       = var_mean[indices_chosen,:,:]
    sst_mean       = sst_mean[indices_chosen,:,:]
    zg_500hPa_mean = zg_500hPa_mean[indices_chosen,:,:]
    years          = years[indices_chosen]
    palmyra_years_with_data_inrange = palmyra_years_with_data[(palmyra_years_with_data>=year_bounds[0]) & (palmyra_years_with_data<=year_bounds[1]) & (palmyra_years_with_data>=years[0]) & (palmyra_years_with_data<=years[-1])]
    #
    # Find the indices of years with/without Palmyra data.
    LMR_indices_palmyra = [years.tolist().index(year) for year in palmyra_years_with_data_inrange]
    LMR_indices_no_palmyra = list(set(range(len(years))) - set(LMR_indices_palmyra))
    #
    # If specified, take either years with Palmyra data or years without
    if year_filter == "only palmyra":
        var_mean       = var_mean[LMR_indices_palmyra,:,:]
        sst_mean       = sst_mean[LMR_indices_palmyra,:,:]
        zg_500hPa_mean = zg_500hPa_mean[LMR_indices_palmyra,:,:]
        years          = years[LMR_indices_palmyra]
    elif year_filter == "no palmyra":
        var_mean       = var_mean[LMR_indices_no_palmyra,:,:]
        sst_mean       = sst_mean[LMR_indices_no_palmyra,:,:]
        zg_500hPa_mean = zg_500hPa_mean[LMR_indices_no_palmyra,:,:]
        years          = years[LMR_indices_no_palmyra]
    #
    # Remove the mean of all years.
    var_mean       = var_mean       - np.mean(var_mean,axis=0)
    sst_mean       = sst_mean       - np.mean(sst_mean,axis=0)
    zg_500hPa_mean = zg_500hPa_mean - np.mean(zg_500hPa_mean,axis=0)
    #
    # Compute average PDSI for the U.S. and the four regions used in Cook et al. 2014.
    # Also, Compute means over the approximate California region (for comparison with Seager et al. 2015).
    # For a better comparison, find a less crude way of averaging over California.
    var_regional_mean = {}
    for region in regions:
        print(region)
        lat_min = region_bounds[region][0]
        lat_max = region_bounds[region][1]
        lon_min = region_bounds[region][2]
        lon_max = region_bounds[region][3]
        var_regional_mean[region] = compute_regional_means.compute_means(var_mean,lat,lon,lat_min,lat_max,lon_min,lon_max)
    #
    # For every region, compute the correlation between drought in that region and SSTs and 500hPa heights.
    if corr_calc_type == 'pearsonr':
        correlations_sst,      _,lat_not_sig_sst,      lon_not_sig_sst       = correlation_calc(var_regional_mean,sst_mean,      lat,lon)
        correlations_zg_500hPa,_,lat_not_sig_zg_500hPa,lon_not_sig_zg_500hPa = correlation_calc(var_regional_mean,zg_500hPa_mean,lat,lon)
    if corr_calc_type == 'isospectral':
        #
        # Set up dictionaries
        correlations_sst = {}
        lat_not_sig_sst  = {}
        lon_not_sig_sst  = {}
        correlations_zg_500hPa = {}
        lat_not_sig_zg_500hPa  = {}
        lon_not_sig_zg_500hPa  = {}
        #
        # Set options
        options = SET(nsim=1000,method='isospectral',alpha=0.05)
        #
        for region in regions:
            #
            # Compute the significance
            starttime = time.time()
            correlations_sst[region],      lat_not_sig_sst[region],      lon_not_sig_sst[region]       = corr_2d_ttest(sst_mean,      var_regional_mean[region],lat,lon,options,1)
            correlations_zg_500hPa[region],lat_not_sig_zg_500hPa[region],lon_not_sig_zg_500hPa[region] = corr_2d_ttest(zg_500hPa_mean,var_regional_mean[region],lat,lon,options,1)
            endtime = time.time()
            print('Time for isospectral calculation: '+str('%1.2f' % ((endtime-starttime)/60))+' minutes')
    #
    # Remove points over land for sst
    lat_not_sig_sst,lon_not_sig_sst = remove_points_over_land(sst_mean,lat_not_sig_sst,lon_not_sig_sst,lat,lon)
    #
    #
    ### FIGURES
    plt.style.use('ggplot')
    #
    # Map
    m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c')
    lon_2d,lat_2d = np.meshgrid(lon,lat)
    x, y = m(lon_2d,lat_2d)
    letters = ['a','e','b','f','c','g','d','h']
    #
    # Plot the correlation between the region of interest and the selected variable everywhere.
    f, ax = plt.subplots(len(regions),2,figsize=(14,14))
    ax = ax.ravel()
    #
    for var_num,variable in enumerate(['SST','500hPa heights']):
        if variable == 'SST':
            correlations_selected = correlations_sst
            lat_not_sig_selected  = lat_not_sig_sst
            lon_not_sig_selected  = lon_not_sig_sst
        elif variable == '500hPa heights':
            correlations_selected = correlations_zg_500hPa
            lat_not_sig_selected  = lat_not_sig_zg_500hPa
            lon_not_sig_selected  = lon_not_sig_zg_500hPa
        #
        # Set max and min values
        if (dataset == 'LMR NAonly/NAexclude') and (year_bounds == [1001,2000]): extreme_r = .2
        else:                                                                    extreme_r = 1
        #
        levels_r = np.linspace(-1*extreme_r,extreme_r,21)
        #
        for i,region in enumerate(regions):
            #
            panel = (i*2)+var_num
            #
            ax[panel].set_title(letters[panel]+") "+region+" region, "+variable,fontsize=20,loc='left')
            m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c',ax=ax[panel])
            if extreme_r == 1: image1 = m.contourf(x,y,correlations_selected[region],levels_r,cmap='RdBu_r',vmin=-1*extreme_r,vmax=extreme_r)
            else:              image1 = m.contourf(x,y,correlations_selected[region],levels_r,extend='both',cmap='RdBu_r',vmin=-1*extreme_r,vmax=extreme_r)
            #
            x2, y2 = m(lon_not_sig_selected[region],lat_not_sig_selected[region])
            m.plot(x2,y2,'ko',markersize=.5)
            cb = m.colorbar(image1,ax=ax[panel],location='bottom').set_label("Correlation",fontsize=18)
            m.drawparallels([0],labels=[True])
            m.drawcoastlines()
            #
            # Draw a box for the region of interest
            lat_min = region_bounds[region][0]
            lat_max = region_bounds[region][1]
            lon_min = region_bounds[region][2]
            lon_max = region_bounds[region][3]
            if lon_min < 0: lon_min = lon_min+360
            if lon_max < 0: lon_max = lon_max+360
            x_region,y_region = m([lon_min,lon_min,lon_max,lon_max],[lat_min,lat_max,lat_max,lat_min])
            xy_region = np.column_stack((x_region,y_region))
            region_box = Polygon(xy_region,edgecolor='black',facecolor='none',linewidth=2,alpha=.5)
            plt.gca().add_patch(region_box)
    #
#    f.suptitle("Correlations between regional drought and spatial climate everywhere, "+dataset+", years "+str(year_bounds[0])+"-"+str(year_bounds[1]),fontsize=20)
    f.tight_layout()
#    f.subplots_adjust(top=.93)
    if save_instead_of_plot == True:
        plt.savefig("figures/correlation_map_"+dataset.replace(" ","_").replace("/","_")+"_"+reference_variable+"_years_"+str(year_bounds[0])+"-"+str(year_bounds[1])+"_"+year_filter.replace(" ","_")+".png",dpi=300,format='png')
        plt.close()
    else:
        plt.show()
    #
    # Put the desired variables together into a single dictionary
    correlation_variables = {}
    correlation_variables['correlations_sst']       = correlations_sst
    correlation_variables['lat_not_sig_sst']        = lat_not_sig_sst
    correlation_variables['lon_not_sig_sst']        = lon_not_sig_sst
    correlation_variables['correlations_zg_500hPa'] = correlations_zg_500hPa
    correlation_variables['lat_not_sig_zg_500hPa']  = lat_not_sig_zg_500hPa
    correlation_variables['lon_not_sig_zg_500hPa']  = lon_not_sig_zg_500hPa
    correlation_variables['lat']                    = lat
    correlation_variables['lon']                    = lon
    correlation_variables['dataset']                = dataset
    correlation_variables['year_bounds']            = year_bounds
    correlation_variables['sst_mean']               = sst_mean
    correlation_variables['var_regional_mean']      = var_regional_mean
    #
    # Return the calculated values
    return correlation_variables

# Make a function that calculates correlations, then makes a figure as well as outputing the variables
def combined_maps(correlation_variables_prod,correlation_variables_prior,correlation_variables_na_full,correlation_variables_na_short):
    #
    # Import global variables
    global calc_bounds
    #
    # Combine into a single dictionary
    correlation_variables_all = {}
    correlation_variables_all[0] = correlation_variables_prod
    correlation_variables_all[1] = correlation_variables_prior
    correlation_variables_all[2] = correlation_variables_na_full
    correlation_variables_all[3] = correlation_variables_na_short
    #
    region = 'Southwest US'
    #
    ### FIGURES
    plt.style.use('ggplot')
    #
    # Plot the correlation between the region of interest and the selected variable everywhere.
    n_exp = len(correlation_variables_all.keys())
    f, ax = plt.subplots(n_exp,2,figsize=(14,14))
    ax = ax.ravel()
    #
    # Map
    for exp_num in range(n_exp):
        #
        correlations_sst       = correlation_variables_all[exp_num]['correlations_sst']
        correlations_zg_500hPa = correlation_variables_all[exp_num]['correlations_zg_500hPa']
        lat_not_sig_sst        = correlation_variables_all[exp_num]['lat_not_sig_sst']
        lon_not_sig_sst        = correlation_variables_all[exp_num]['lon_not_sig_sst']
        lat_not_sig_zg_500hPa  = correlation_variables_all[exp_num]['lat_not_sig_zg_500hPa']
        lon_not_sig_zg_500hPa  = correlation_variables_all[exp_num]['lon_not_sig_zg_500hPa']
        lat                    = correlation_variables_all[exp_num]['lat']
        lon                    = correlation_variables_all[exp_num]['lon']
        dataset                = correlation_variables_all[exp_num]['dataset']
        year_bounds            = correlation_variables_all[exp_num]['year_bounds']
        #
        m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c')
        lon_2d,lat_2d = np.meshgrid(lon,lat)
        x, y = m(lon_2d,lat_2d)
        letters = ['a','b','c','d','e','f','g','h']
        #
        for var_num,variable in enumerate(['SST','500hPa heights']):
            if variable == 'SST':
                correlations_selected = correlations_sst
                lat_not_sig_selected  = lat_not_sig_sst
                lon_not_sig_selected  = lon_not_sig_sst
            elif variable == '500hPa heights':
                correlations_selected = correlations_zg_500hPa
                lat_not_sig_selected  = lat_not_sig_zg_500hPa
                lon_not_sig_selected  = lon_not_sig_zg_500hPa
            #
            # Set max and min values
            if (dataset == 'LMR NAonly/NAexclude') and (year_bounds == [1001,2000]): extreme_r = .2
            else:                                                                    extreme_r = 1
            #
            levels_r = np.linspace(-1*extreme_r,extreme_r,21)
            panel = (exp_num*2)+var_num
            #
            ax[panel].set_title(letters[panel]+") "+variable+", "+dataset+", years "+str(year_bounds[0])+"-"+str(year_bounds[1]),fontsize=13.5,loc='left')
            m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c',ax=ax[panel])
            if extreme_r == 1: image1 = m.contourf(x,y,correlations_selected[region],levels_r,cmap='RdBu_r',vmin=-1*extreme_r,vmax=extreme_r)
            else:              image1 = m.contourf(x,y,correlations_selected[region],levels_r,extend='both',cmap='RdBu_r',vmin=-1*extreme_r,vmax=extreme_r)
            #
            x2, y2 = m(lon_not_sig_selected[region],lat_not_sig_selected[region])
            m.plot(x2,y2,'ko',markersize=.5)
            cb = m.colorbar(image1,ax=ax[panel],location='bottom').set_label("Correlation",fontsize=18)
            m.drawparallels([0],labels=[True])
            m.drawcoastlines()
            #
            # Draw a box for the region of interest
            lat_min = region_bounds[region][0]
            lat_max = region_bounds[region][1]
            lon_min = region_bounds[region][2]
            lon_max = region_bounds[region][3]
            if lon_min < 0: lon_min = lon_min+360
            if lon_max < 0: lon_max = lon_max+360
            x_region,y_region = m([lon_min,lon_min,lon_max,lon_max],[lat_min,lat_max,lat_max,lat_min])
#            xy_region = zip(x_region,y_region)
            xy_region = np.column_stack((x_region,y_region))
            region_box = Polygon(xy_region,edgecolor='black',facecolor='none',linewidth=2,alpha=.5)
            plt.gca().add_patch(region_box)
    #
#    f.suptitle("Correlations between regional drought and spatial climate in the southwest U.S.",fontsize=20)
    f.tight_layout()
#    f.subplots_adjust(top=.93)
    if save_instead_of_plot == True:
        plt.savefig("figures/correlation_maps_southwest_us_all.png",dpi=300,format='png')
        plt.close()
    else:
        plt.show()

# Calculate using the parameters specified at the top of the script.
correlation_variables_prod     = correlation_maps('LMR posterior',       scpdsi_mean_prod_selected, sst_mean_prod_selected, zg_500hPa_mean_prod_selected, lat_prod_selected, lon_prod_selected, years_prod, [1001,2000])
correlation_variables_prior    = correlation_maps('CCSM4 prior',         scpdsi_mean_prior_selected,sst_mean_prior_selected,zg_500hPa_mean_prior_selected,lat_prior_selected,lon_prior_selected,years_prior,[850,1850])
correlation_variables_na_full  = correlation_maps('LMR NAonly/NAexclude',scpdsi_mean_na_selected,   sst_mean_na_selected,   zg_500hPa_mean_na_selected,   lat_na_selected,   lon_na_selected,   years_na,   [1001,2000])
correlation_variables_na_short = correlation_maps('LMR NAonly/NAexclude',scpdsi_mean_na_selected,   sst_mean_na_selected,   zg_500hPa_mean_na_selected,   lat_na_selected,   lon_na_selected,   years_na,   [1950,1997])

# Make a combined figure of all of the southwest U.S. correlations
combined_maps(correlation_variables_prod,correlation_variables_prior,correlation_variables_na_full,correlation_variables_na_short)


"""
# Calculate maps for different subsets of years.
for year_filter_selected in ['all years','only palmyra','no palmyra']:
    correlation_maps(dataset,scpdsi_mean,pr_mean,sst_mean,zg_500hPa_mean,nino34_mean,soi_mean,pdo_mean,years,regions,region_bounds,save_instead_of_plot,remove_influence_of_nino34,remove_influence_of_pdo,reference_variable,[1001,2000],year_filter_selected)
"""

"""
# Calculate climate patterns for every century between 1001 and 2000.
for i in range(10):
    year_begin = (i*100)+1001
    year_end = year_begin+99
    print "Calculating extreme events, years "+str(year_begin)+"-"+str(year_end)
    correlation_maps(data_dir,experiment_name1,experiment_name2,True,False,False,reference_variable,'southwest',[year_begin,year_end],year_filter)
"""

"""
plt.contourf(correlations_sst['Central US'][60:68,140:])
plt.colorbar()

values = correlations_sst['Central US'][60:68,140:].flatten()
np.nanmax(values)
"""

"""
plt.contourf(correlations_sst['Southwest US'][35:55, 100:140])
plt.colorbar()

values = correlations_sst['Southwest US'][35:55, 100:140].flatten()
np.nanmin(values)
"""
