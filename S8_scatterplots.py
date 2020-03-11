#=============================================================================
# This script makes comparisons between drought in the southwest U.S. and
# NINO3.4 in the LMR and in data.
#    author: Michael P. Erb
#    date  : 1/7/2019
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import compute_regional_means
from scipy import stats
import seaborn as sns
import scipy.io
import mpe_functions as mpe


save_instead_of_plot = True
data_dir = '/projects/pd_lab/data/LMR/archive_output/'

# If desired, specify a lead for the Nino3.4 data.
var1_lead = 0

# Specify the bounds of extra regions
region_bounds = {}
region_bounds['Southwest US']     = [32,  40,-125,  -105]  # 32-40N, 125-105W
region_bounds['Southwest US new'] = [28,  40,-115,   -95]  # 28-40N, 115-95W
region_bounds['California']       = [32,  42,-122,  -114]  # 32-42N, 122-114W
region_bounds['Monsoon']          = [28,  38,-110,  -104]  # 28-38N, 110-104W

possible_regions = ['Southwest US']


# Use the mean or subsampled data.
data_to_use = 'mean'
#data_to_use = 'subsample'

# Sometimes, it's useful to do this analysis using two experiments.  If that's the case, specify them here.
experiment_name1 = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
experiment_name2 = experiment_name1
#experiment_name1  = 'NorthAmerica_only_r0_r4_temporary'
#experiment_name2  = 'NorthAmerica_exclude_r0_r4_temporary'

# Years of interest
year_bounds = [1001,2000]
#year_bounds = [1950,1997]
#year_bounds = [1950,1997]  # Three corals overlap
#year_bounds = [1886,1997]  # Palmyra segment 1
#year_bounds = [1635,1702]  # Palmyra segment 2
#year_bounds = [1317,1464]  # Palmyra segment 3
#boundary_year = 1851
#boundary_year = 1873
boundary_year = 1874
extreme_percentile = 15


### LOAD DATA

# Load LMR data
handle = xr.open_dataset(data_dir+experiment_name1+'/pdsi_MCruns_ensemble_'+data_to_use+'.nc',decode_times=False)
scpdsi_lmr_all = handle['pdsi'].values
lat_lmr        = handle['lat'].values
lon_lmr        = handle['lon'].values
time_lmr       = handle['time'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name1+'/prate_MCruns_ensemble_'+data_to_use+'.nc',decode_times=False)
pr_lmr_all = handle['prate'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2+'/air_MCruns_ensemble_'+data_to_use+'.nc',decode_times=False)
tas_lmr_all = handle['air'].values
handle.close()

#TODO: Make climate index calculations for the means
if data_to_use == 'mean':  
    handle = xr.open_dataset(data_dir+experiment_name2+'/climate_indices_MCruns_ensemble_'+data_to_use+'_calc_from_posterior.nc',decode_times=False)
    nino34_lmr_all = handle['nino34'].values
    soi_lmr_all    = handle['soi'].values
    amo_lmr_all    = handle['amo'].values
    pdo_lmr_all    = handle['pdo'].values
    handle.close()
elif data_to_use == 'subsample':
    data_dir_new = '/projects/pd_lab/data/LMR/archive_output/production_indices/'
    handle_new = xr.open_dataset(data_dir_new+'/posterior_climate_indices_MCruns_ensemble_subsample.nc',decode_times=False)
    nino34_lmr_all = handle_new['nino34'].values
    soi_lmr_all    = handle_new['soi'].values
    amo_lmr_all    = handle_new['amo'].values
    pdo_lmr_all    = handle_new['pdo'].values
    handle_new.close()

years_lmr = time_lmr/365
years_lmr = years_lmr.astype(int)


# If the subsampled data are used, reshape the array so that iterations and ensemble members are on the same axis.
if data_to_use == 'subsample':
    #
    # Get dimensions
    ntime = scpdsi_lmr_all.shape[0]
    niter = scpdsi_lmr_all.shape[1]
    nlat  = scpdsi_lmr_all.shape[2]
    nlon  = scpdsi_lmr_all.shape[3]
    nens  = scpdsi_lmr_all.shape[4]
    #
    # Roll axes so that iterations and ensemble members are next to each other
    scpdsi_lmr_rolled = np.rollaxis(scpdsi_lmr_all,4,2)
    pr_lmr_rolled     = np.rollaxis(pr_lmr_all,    4,2)
    tas_lmr_rolled    = np.rollaxis(tas_lmr_all,   4,2)
    #
    # Reshape iterations and ensemble members to be on the same axis
    scpdsi_lmr_all = np.reshape(scpdsi_lmr_rolled,(ntime,niter*nens,nlat,nlon))
    pr_lmr_all     = np.reshape(pr_lmr_rolled,    (ntime,niter*nens,nlat,nlon))
    tas_lmr_all    = np.reshape(tas_lmr_rolled,   (ntime,niter*nens,nlat,nlon))
    #
    # Also reshape the climate index variables
    nino34_lmr_all = np.reshape(nino34_lmr_all,(ntime,niter*nens))
    soi_lmr_all    = np.reshape(soi_lmr_all,   (ntime,niter*nens))
    amo_lmr_all    = np.reshape(amo_lmr_all,   (ntime,niter*nens))
    pdo_lmr_all    = np.reshape(pdo_lmr_all,   (ntime,niter*nens))



# Load DaiPDSI data set
handle_dai = xr.open_dataset('/projects/pd_lab/data/LMR/data/analyses/DaiPDSI/Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc',decode_times=False)
pdsi_dai_monthly = handle_dai['pdsi'].values
lat_dai          = handle_dai['lat'].values
lon_dai          = handle_dai['lon'].values
handle_dai.close()
years_dai = np.arange(1850,2015)

climate_ind_dir = '/projects/pd_lab/data/modern_datasets/climate_indices/'

# Load Bunge & Clark Nino3.4 data - Data runs from January 1873 to March 2008
nino34_obs = scipy.io.loadmat(climate_ind_dir+'Nino34/Bunge_and_Clarke_2009/NINO34.mat')
nino34_data_1d = np.squeeze(nino34_obs['nino34'])
nino34_data = np.reshape(nino34_data_1d[0:1620],(135,12))
nino34_years = np.arange(1873,2008)


### CALCULATIONS

# Compute annual mean values for the DaiPDSI and Bunge & Clark data sets

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

# Bunge & Clark data runs from January 1873 to March 2008
nino34_data_annual = mpe.annual_mean(nino34_years,nino34_data)

# Change LMR precipitation to mm/day
pr_lmr_all = pr_lmr_all*60*60*24

# Mask the PDSI
scpdsi_lmr_all[scpdsi_lmr_all == 0] = np.nan
scpdsi_lmr_all  = ma.masked_invalid(scpdsi_lmr_all)
pdsi_dai_annual = ma.masked_invalid(pdsi_dai_annual)

# Compute a mean over the chosen region
nyears_lmr = scpdsi_lmr_all.shape[0]
niter_lmr  = scpdsi_lmr_all.shape[1]
scpdsi_lmr_mean = {}
pr_lmr_mean     = {}
tas_lmr_mean    = {}
for region in possible_regions:
    scpdsi_lmr_mean[region] = np.zeros((nyears_lmr,niter_lmr)); scpdsi_lmr_mean[region][:] = np.nan
    pr_lmr_mean[region]     = np.zeros((nyears_lmr,niter_lmr)); pr_lmr_mean[region][:]     = np.nan
    tas_lmr_mean[region]    = np.zeros((nyears_lmr,niter_lmr)); tas_lmr_mean[region][:]    = np.nan
    #
    print('Averaging over '+region)
    lat_min = region_bounds[region][0]
    lat_max = region_bounds[region][1]
    lon_min = region_bounds[region][2]
    lon_max = region_bounds[region][3]
    for i in range(niter_lmr):
        print(' === Computing means for iteration '+str(i+1)+'/'+str(niter_lmr)+' ===')
        scpdsi_lmr_mean[region][:,i] = compute_regional_means.compute_means(scpdsi_lmr_all[:,i,:,:],lat_lmr,lon_lmr,lat_min,lat_max,lon_min,lon_max)
        pr_lmr_mean[region][:,i]     = compute_regional_means.compute_means(pr_lmr_all[:,i,:,:],    lat_lmr,lon_lmr,lat_min,lat_max,lon_min,lon_max)
        tas_lmr_mean[region][:,i]    = compute_regional_means.compute_means(tas_lmr_all[:,i,:,:],   lat_lmr,lon_lmr,lat_min,lat_max,lon_min,lon_max)

# Compute means for the DaiPDSI data set as well.
pdsi_dai_mean = {}
for region in possible_regions:
    #
    print('Averaging over '+region)
    lat_min = region_bounds[region][0]
    lat_max = region_bounds[region][1]
    lon_min = region_bounds[region][2]
    lon_max = region_bounds[region][3]
    #
    pdsi_dai_mean[region] = compute_regional_means.compute_means(pdsi_dai_annual,lat_dai,lon_dai,lat_min,lat_max,lon_min,lon_max)


# Shorten both observational datasets to 1873-2000
year_start = 1873
year_end   = 2000
years_obs  = np.arange(year_start,year_end+1)

index1_dai = np.where(years_dai == year_start)[0][0]
index2_dai = np.where(years_dai == year_end)[0][0]
years_dai_shortened = years_dai[index1_dai:index2_dai+1]
pdsi_dai_mean_shortened = {}
#for region in pdsi_dai_mean.keys():
for region in possible_regions:
    pdsi_dai_mean_shortened[region] = pdsi_dai_mean[region][index1_dai:index2_dai+1]

index1_nino34 = np.where(nino34_years == year_start)[0][0]
index2_nino34 = np.where(nino34_years == year_end)[0][0]
nino34_years_shortened = nino34_years[index1_nino34:index2_nino34+1]
nino34_data_annual_shortened = nino34_data_annual[index1_nino34:index2_nino34+1]


# Make a scatterplot of the data
"""
var1,var2,var1_name,var2_name,years_obs,nino34_obs,pdsi_obs,time_mean,years_lmr,year_bounds,boundary_year,data_to_use,var1_lead,title_text,save_instead_of_plot = \
    nino34_lmr_all,scpdsi_lmr_mean['Southwest US new'],'Nino3.4','Southwest U.S. PDSI new',years_obs,nino34_data_annual_shortened,\
    pdsi_dai_mean_shortened['Southwest US new'],'annual',years_lmr,year_bounds,boundary_year,data_to_use,var1_lead,'testing',save_instead_of_plot
"""

def make_scatterplot(var1,var2,var1_name,var2_name,years_obs,nino34_obs,pdsi_obs,time_mean,years_lmr,year_bounds,boundary_year,data_to_use,var1_lead,title_text,save_instead_of_plot):
    #
    print("Making scatterplot for "+var1_name+" vs. "+var2_name+", "+time_mean+" time-scale.")
    #
    # Select data
    idx_begin = np.where(years_lmr == year_bounds[0])[0][0]
    idx_end   = np.where(years_lmr == year_bounds[1])[0][0]
    var1_selected  = var1[idx_begin-var1_lead:idx_end+1-var1_lead,:]
    var2_selected  = var2[idx_begin:idx_end+1,:]
    years_selected = years_lmr[idx_begin:idx_end+1]
    #
    # Remove the means of the entire period
    var1_selected = var1_selected - np.mean(var1_selected,axis=0)
    var2_selected = var2_selected - np.mean(var2_selected,axis=0)
    #
    # Find the boundary index for the shortened data set.
    boundary_year = boundary_year
    idx_boundary = np.where(years_selected == boundary_year)[0][0]
    #
    # If desired, shift observational data to make Nino3.4 lead PDSI
    if var1_lead == 0:
        years_obs  = years_obs[1:]
        pdsi_obs   = pdsi_obs[1:]
        nino34_obs = nino34_obs[1:]
    elif var1_lead != 0:
        years_obs  = years_obs[var1_lead:]
        pdsi_obs   = pdsi_obs[var1_lead:]
        nino34_obs = nino34_obs[0:-1*var1_lead]
    #
    # If selected, compute decadal means
    if time_mean == 'decadal':
        nyears = var1_selected.shape[0]
        niter  = var1_selected.shape[1]
        var1_selected = np.mean(np.reshape(var1_selected, (int(nyears/10),10,niter)),axis=1)
        var2_selected = np.mean(np.reshape(var2_selected, (int(nyears/10),10,niter)),axis=1)
        years_decadal = np.mean(np.reshape(years_selected,(int(nyears/10),10)),axis=1)
        idx_boundary = np.abs(years_decadal-boundary_year).argmin()
        #
        if (var1_name == 'Nino3.4') and ((var2_name == 'Southwest U.S. PDSI') or (var2_name == 'Southwest U.S. PDSI new')):
            idx_obs_begin = np.where(years_obs == 1881)[0][0]
            idx_obs_end   = np.where(years_obs == 2000)[0][0]
            nino34_obs_selected = nino34_obs[idx_obs_begin:idx_obs_end+1]
            pdsi_obs_selected   = pdsi_obs[idx_obs_begin:idx_obs_end+1]
            years_obs_selected  = years_obs[idx_obs_begin:idx_obs_end+1]
            nyears_obs = len(years_obs_selected)
            #
            nino34_obs = np.mean(np.reshape(nino34_obs_selected,(int(nyears_obs/10),10)),axis=1)
            pdsi_obs   = np.mean(np.reshape(pdsi_obs_selected,  (int(nyears_obs/10),10)),axis=1)
    #
    # Create versions of the data for the entire data set, as well as before and after the boundary
    var1_selected_1d_all    = var1_selected.flatten()
    var2_selected_1d_all    = var2_selected.flatten()
    var1_selected_1d_before = var1_selected[0:idx_boundary,:].flatten()
    var2_selected_1d_before = var2_selected[0:idx_boundary,:].flatten()
    var1_selected_1d_after  = var1_selected[idx_boundary:,:].flatten()
    var2_selected_1d_after  = var2_selected[idx_boundary:,:].flatten()
    #
    # Calculate some statistics
    var2_var1_below_average      = var2_selected_1d_all[var1_selected_1d_all < 0]
    var2_var1_below_10percentile = var2_selected_1d_all[var1_selected_1d_all < np.percentile(var1_selected_1d_all,10)]
    fraction1 = sum(var2_var1_below_average < 0) / float(var2_var1_below_average.shape[0])
    fraction2 = sum(var2_var1_below_10percentile < 0) / float(var2_var1_below_10percentile.shape[0])
    print(' == Statistics ==')
    print('Percentage of years (or decades) with a below-average '+var1_name+' which also have a below-average '+var2_name+': '+str(fraction1))
    print('Percentage of years (or decades) with a below-10-percentile '+var1_name+' which also have a below-average '+var2_name+': '+str(fraction2))
    #
    # Calculate regression statistics
    slope,       intercept,       r_value,       p_value,       std_err        = stats.linregress(var1_selected_1d_all,   var2_selected_1d_all)
    slope_before,intercept_before,r_value_before,p_value_before,std_err_before = stats.linregress(var1_selected_1d_before,var2_selected_1d_before)
    slope_after, intercept_after, r_value_after, p_value_after, std_err_after  = stats.linregress(var1_selected_1d_after, var2_selected_1d_after)
    if (var1_name == 'Nino3.4') and ((var2_name == 'Southwest U.S. PDSI') or (var2_name == 'Southwest U.S. PDSI new')):
        slope_obs,intercept_obs,r_value_obs,p_value_obs,std_err_obs = stats.linregress(nino34_obs,pdsi_obs)
    #
    # Find the indices of the extreme values of var1 and var2
    idx_low1  = np.where(var1_selected_1d_all < np.percentile(var1_selected_1d_all,extreme_percentile))[0]
    idx_high1 = np.where(var1_selected_1d_all > np.percentile(var1_selected_1d_all,100-extreme_percentile))[0]
    #
    idx_low2  = np.where(var2_selected_1d_all < np.percentile(var2_selected_1d_all,extreme_percentile))[0]
    idx_high2 = np.where(var2_selected_1d_all > np.percentile(var2_selected_1d_all,100-extreme_percentile))[0]
    #
    #
    ### FIGURES
    plt.style.use('ggplot')
    #
    # FIGURE - Comparing two variables.
    #
    # Specify the bins for the histograms
    var1_min = np.min(var1_selected_1d_all)
    var1_max = np.max(var1_selected_1d_all)
    var1_range = np.linspace(var1_min,var1_max,50)
    var2_min = np.min(var2_selected_1d_all)
    var2_max = np.max(var2_selected_1d_all)
    var2_range = np.linspace(var2_min,var2_max,50)
    #
    plt.figure(figsize=(20,20))
    #
    # Plot the values before the cutoff year
    plot = sns.JointGrid(x=var1_selected_1d_before,y=var2_selected_1d_before)
    plot = plot.plot_joint(plt.scatter,c='k',label="LMR, "+str(year_bounds[0])+"-"+str(boundary_year-1)+" C.E. (R$^2$="+str('%1.2f' % r_value_before**2)+")",alpha=0.5)
    #
    # Plot the values after the cutoff year
    plot.x = var1_selected_1d_after
    plot.y = var2_selected_1d_after
    plot = plot.plot_joint(plt.scatter,c='green',label="LMR, "+str(boundary_year)+"-"+str(year_bounds[1])+" C.E. (R$^2$="+str('%1.2f' % r_value_after**2)+")",alpha=0.5)
    #
    # Plot the obs
    if (var1_name == 'Nino3.4') and ((var2_name == 'Southwest U.S. PDSI') or (var2_name == 'Southwest U.S. PDSI new')):
        plot.x = nino34_obs
        plot.y = pdsi_obs
        plot = plot.plot_joint(plt.scatter,c='yellow',label="Obs., "+str(years_obs[0])+"-"+str(years_obs[-1])+" C.E. (R$^2$="+str('%1.2f' % r_value_obs**2)+")",alpha=0.5)
    #
    # Plot histograms on the margins
    plot.ax_marg_x.hist(var1_selected_1d_all[idx_high2],bins=var1_range,alpha=0.5)
    plot.ax_marg_x.hist(var1_selected_1d_all[idx_low2],bins=var1_range,alpha=0.5)
    plot.ax_marg_y.hist(var2_selected_1d_all[idx_high1],bins=var2_range,orientation='horizontal',alpha=0.5)
    plot.ax_marg_y.hist(var2_selected_1d_all[idx_low1],bins=var2_range,orientation='horizontal',alpha=0.5)
    #
    # Add panel labels
    plt.text(.03,.93,'a)',transform=plot.ax_joint.transAxes,fontsize=16)
    plt.text(.03,1.1,'b)',transform=plot.ax_marg_x.transAxes,fontsize=16)
    plt.text(.1,1.02,'c)',transform=plot.ax_marg_y.transAxes,fontsize=16)
    #
    plt.legend(loc=1,bbox_to_anchor=(1.32,-.06),prop={'size':11})
    plt.xlabel(var1_name)
    plt.ylabel(var2_name)
    plt.subplots_adjust(top=0.9)
    plot.fig.suptitle(var1_name+" vs. "+var2_name+" "+time_mean+"-mean anomalies.\nLMR R$^2$="+str('%1.2f' % r_value**2))
    #
    if save_instead_of_plot:
        plt.savefig("figures/scatterplot_v2_"+time_mean+"_"+var1_name.replace(' ','').replace('.','')+"_"+var2_name.replace(' ','').replace('.','')+"_"+str(year_bounds[0])+"_"+str(year_bounds[1])+"_"+data_to_use+"_ninolead_"+str(var1_lead)+".png",bbox_inches="tight",dpi=300,format='png')
    else:
        plt.show()


# Make some scatterplots - one with no lead and one with a 1-year lead.
chosen_region = 'Southwest US'
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'annual',years_lmr,year_bounds,boundary_year,data_to_use,0,'southwest U.S. region',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'annual',years_lmr,year_bounds,boundary_year,data_to_use,1,'southwest U.S. region, 1-year Nino3.4 lead',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'decadal',years_lmr,year_bounds,boundary_year,data_to_use,0,'southwest U.S. region',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'decadal',years_lmr,year_bounds,boundary_year,data_to_use,1,'southwest U.S. region, 1-year Nino3.4 lead',save_instead_of_plot)

"""
chosen_region = 'Southwest US new'
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI new',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'annual',years_lmr,year_bounds,boundary_year,data_to_use,0,'new southwest U.S. region',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI new',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'annual',years_lmr,year_bounds,boundary_year,data_to_use,1,'new southwest U.S. region, 1-year Nino3.4 lead',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI new',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'decadal',years_lmr,year_bounds,boundary_year,data_to_use,0,'new southwest U.S. region',save_instead_of_plot)
make_scatterplot(nino34_lmr_all,scpdsi_lmr_mean[chosen_region],'Nino3.4','Southwest U.S. PDSI new',years_obs,nino34_data_annual_shortened,pdsi_dai_mean_shortened[chosen_region],'decadal',years_lmr,year_bounds,boundary_year,data_to_use,1,'new southwest U.S. region, 1-year Nino3.4 lead',save_instead_of_plot)
"""

#print(np.square(np.corrcoef(nino34_lmr_all[1874:,:].flatten(),scpdsi_lmr_mean[chosen_region][1874:,:].flatten())[0,1]))
#print(np.square(np.corrcoef(np.mean(nino34_lmr_all[1874:,:],axis=1),np.mean(scpdsi_lmr_mean[chosen_region][1874:,:],axis=1))[0,1]))


