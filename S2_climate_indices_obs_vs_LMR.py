#=============================================================================
# This script compares several quantitles from the LMR against observations.
# Data sources:
#    * Nino 3.4: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/  (Right?)
#    * PDO:      http://research.jisao.washington.edu/pdo/
#    * AMO:      http://www.esrl.noaa.gov/psd/data/timeseries/AMO/
#    * SOI:      https://www.ncdc.noaa.gov/teleconnections/enso/indicators/soi/
#
# In additional to the already-computed indices, AMO is calculated in a way to
# match the observationally-based AMO dataset at the link above.
#
#    author: Michael P. Erb
#    date  : 5/29/2018
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import xarray as xr
import mpe_functions as mpe
import scipy.io
import scipy

save_instead_of_plot = True

data_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
climate_ind_dir = '/projects/pd_lab/data/modern_datasets/climate_indices/'


### LOAD DATA

# Load  Nino3.4 data - Data runs from January 1873 to March 2008
nino34_obs = scipy.io.loadmat(climate_ind_dir+'Nino34/Bunge_and_Clarke_2009/NINO34.mat')
nino34_data_1d = np.squeeze(nino34_obs['nino34'])
nino34_data = np.reshape(nino34_data_1d[0:1620],(135,12))
nino34_years = np.arange(1873,2008)

pdo_obs = np.loadtxt(climate_ind_dir+'PDO/PDO.latest.edited_for_input.txt',skiprows=32)
amo_obs = np.loadtxt(climate_ind_dir+'AMO/amon.us.long.edited_for_input.data',skiprows=4)
soi_obs = np.genfromtxt(climate_ind_dir+'SOI/data.csv',delimiter=',',skip_header=2)

# Load the LMR indices calcuated with the newer code (Nino3.4 and the old AMO are basically identical)
data_dir_new = '/projects/pd_lab/data/LMR/archive_output/production_indices/'
handle_new = xr.open_dataset(data_dir_new+'/posterior_climate_indices_MCruns_ensemble_subsample.nc',decode_times=False)
nino34_lmr = handle_new['nino34'].values
pdo_lmr    = handle_new['pdo'].values
amo_lmr    = handle_new['amo'].values
soi_lmr    = handle_new['soi'].values
time_lmr   = handle_new['time'].values
handle_new.close()

handle = xr.open_dataset(data_dir+experiment_name+'/sst_MCruns_ensemble_subsample.nc',decode_times=False)
sst_lmr_all = handle['sst'].values
lon_lmr     = handle['lon'].values
lat_lmr     = handle['lat'].values
handle.close()

years_lmr = time_lmr/365
years_lmr = years_lmr.astype(int)


### CALCULATIONS

# Measure some dimensions.
nyears = nino34_lmr.shape[0]
niter  = nino34_lmr.shape[1]
nens   = nino34_lmr.shape[2]

# Calculate an alternate formation of AMO
def calculate_AMO_new( sst,years,lat,lon ):
    #
    # Compute the mean over the North Atlantic region (0-70N, 80W-0)
    j_indices = np.where((lat >= 0) & (lat <= 70))[0]          # All latitudes between 0 and 70N
    i_indices = np.where((lon >= (360-80)) & (lon <= 360))[0]  # All longitudes between 80W and 0W
    j_min = min(j_indices); j_max = max(j_indices)
    i_min = min(i_indices); i_max = max(i_indices)
    sst_mean_NAtl = mpe.spatial_mean(sst,lat,j_min,j_max,i_min,i_max)
    #
    # Remove a linear trend for the time series.
    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(years,sst_mean_NAtl)
    reconstruction = (slope*years) + intercept
    sst_mean_NAtl_detrended = sst_mean_NAtl-reconstruction
    #
    return sst_mean_NAtl_detrended

amo_years_new = np.arange(1856,2001)
nyears_new = len(amo_years_new)
amo_new_lmr = np.zeros((nyears_new,niter,nens)); amo_new_lmr[:] = np.nan

# Select the same years as are used for the instrumentally-based AMO.
index_begin = np.where(years_lmr == amo_years_new[0])[0][0]
index_end   = np.where(years_lmr == amo_years_new[-1])[0][0]
sst_selected = sst_lmr_all[index_begin:index_end+1,:,:,:]

for iteration in range(niter):
    for ens_member in range(nens):
        print('\n === Calculating climate indices.  Iteration: '+str(iteration+1)+'/'+str(niter)+',  Ensemble member: '+str(ens_member+1)+'/'+str(nens)+' ===')
        amo_new_lmr[:,iteration,ens_member] = calculate_AMO_new( sst_selected[:,iteration,:,:,ens_member],amo_years_new,lat_lmr,lon_lmr )


# Put LMR iterations and ensemble members on the same axis
nino34_lmr  = np.reshape(nino34_lmr, (nyears,niter*nens))
pdo_lmr     = np.reshape(pdo_lmr,    (nyears,niter*nens))
amo_lmr     = np.reshape(amo_lmr,    (nyears,niter*nens))
soi_lmr     = np.reshape(soi_lmr,    (nyears,niter*nens))
amo_new_lmr = np.reshape(amo_new_lmr,(nyears_new,niter*nens))

# Compute annual-mean climate indices from the data.
nino34_obs_annual = mpe.annual_mean(nino34_years,nino34_data)

pdo_obs_years = pdo_obs[:,0]
pdo_obs_annual = mpe.annual_mean(pdo_obs_years,pdo_obs[:,1:13])

amo_obs[amo_obs == -99.99] = np.nan
amo_obs_years = amo_obs[:,0]
amo_obs_annual = mpe.annual_mean(amo_obs_years,amo_obs[:,1:13])

soi_obs_years = np.arange(1951,2016)
soi_obs_monthly = np.reshape(soi_obs[0:780,1],(65,12))
soi_obs_annual = mpe.annual_mean(soi_obs_years,soi_obs_monthly[:,0:12])

# Remove the 1951-1980 mean from all records.
nino34_obs_annual  = nino34_obs_annual  - np.mean(nino34_obs_annual[np.where(nino34_years==1951)[0][0]:np.where(nino34_years==1980)[0][0]+1])
pdo_obs_annual     = pdo_obs_annual     - np.mean(pdo_obs_annual[np.where(pdo_obs_years==1951)[0][0]:np.where(pdo_obs_years==1980)[0][0]+1])
amo_obs_annual     = amo_obs_annual     - np.mean(amo_obs_annual[np.where(amo_obs_years==1951)[0][0]:np.where(amo_obs_years==1980)[0][0]+1])
soi_obs_annual     = soi_obs_annual     - np.mean(soi_obs_annual[np.where(soi_obs_years==1951)[0][0]:np.where(soi_obs_years==1980)[0][0]+1])

nino34_lmr  = nino34_lmr  - np.mean(np.mean(nino34_lmr,axis=1)[np.where(years_lmr==1951)[0][0]:np.where(years_lmr==1980)[0][0]+1])
pdo_lmr     = pdo_lmr     - np.mean(np.mean(pdo_lmr,axis=1)[np.where(years_lmr==1951)[0][0]:np.where(years_lmr==1980)[0][0]+1])
amo_lmr     = amo_lmr     - np.mean(np.mean(amo_lmr,axis=1)[np.where(years_lmr==1951)[0][0]:np.where(years_lmr==1980)[0][0]+1])
soi_lmr     = soi_lmr     - np.mean(np.mean(soi_lmr,axis=1)[np.where(years_lmr==1951)[0][0]:np.where(years_lmr==1980)[0][0]+1])
amo_new_lmr = amo_new_lmr - np.mean(np.mean(amo_new_lmr,axis=1)[np.where(amo_years_new==1951)[0][0]:np.where(amo_years_new==1980)[0][0]+1])


# Compute the level of agreement between the data and LMR indices.
def agreement(ts1,ts2):
    correlation = np.corrcoef(ts1,ts2)[0,1]
    R_squared   = (scipy.stats.linregress(ts1,ts2)[2])**2
    CE          = 1 - ( np.sum(np.power(ts1-ts2,2),axis=0) / np.sum(np.power(ts2-np.mean(ts2,axis=0),2),axis=0) )
    #
    return correlation, R_squared, CE

nino34_year_range = [1873,2000]
pdo_year_range    = [1900,2000]
amo_year_range    = [1856,2000]
soi_year_range    = [1951,2000]

nino34_correlation_post,  nino34_R_squared_post,  nino34_CE_post  = agreement(np.mean(nino34_lmr[np.where(years_lmr==nino34_year_range[0])[0][0]:np.where(years_lmr==nino34_year_range[1])[0][0]+1,:],axis=1),   nino34_obs_annual[np.where(nino34_years==nino34_year_range[0])[0][0]:np.where(nino34_years==nino34_year_range[1])[0][0]+1])
pdo_correlation_post,     pdo_R_squared_post,     pdo_CE_post     = agreement(np.mean(pdo_lmr[np.where(years_lmr==pdo_year_range[0])[0][0]:np.where(years_lmr==pdo_year_range[1])[0][0]+1,:],axis=1),            pdo_obs_annual[np.where(pdo_obs_years==pdo_year_range[0])[0][0]:np.where(pdo_obs_years==pdo_year_range[1])[0][0]+1])
amo_correlation_post,     amo_R_squared_post,     amo_CE_post     = agreement(np.mean(amo_lmr[np.where(years_lmr==amo_year_range[0])[0][0]:np.where(years_lmr==amo_year_range[1])[0][0]+1,:],axis=1),            amo_obs_annual[np.where(amo_obs_years==amo_year_range[0])[0][0]:np.where(amo_obs_years==amo_year_range[1])[0][0]+1])
soi_correlation_post,     soi_R_squared_post,     soi_CE_post     = agreement(np.mean(soi_lmr[np.where(years_lmr==soi_year_range[0])[0][0]:np.where(years_lmr==soi_year_range[1])[0][0]+1,:],axis=1),            soi_obs_annual[np.where(soi_obs_years==soi_year_range[0])[0][0]:np.where(soi_obs_years==soi_year_range[1])[0][0]+1])
amo_new_correlation_post, amo_new_R_squared_post, amo_new_CE_post = agreement(np.mean(amo_new_lmr[np.where(amo_years_new==amo_year_range[0])[0][0]:np.where(amo_years_new==amo_year_range[1])[0][0]+1,:],axis=1),amo_obs_annual[np.where(amo_obs_years==amo_year_range[0])[0][0]:np.where(amo_obs_years==amo_year_range[1])[0][0]+1])


### FIGURES
plt.style.use('ggplot')

# FIGURE: Plot Nino3.4, PDO, and AMO (new formulation)
f, ax = plt.subplots(3,1,figsize=(10,10))
ax = ax.ravel()

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

line1, = ax[0].plot(nino34_years,nino34_obs_annual,color='k',linewidth=2)
line2  = ax[0].fill_between(years_lmr,np.percentile(nino34_lmr,2.5,axis=1),np.percentile(nino34_lmr,97.5,axis=1),facecolor='b',alpha=0.2)
line3, = ax[0].plot(years_lmr,np.mean(nino34_lmr,axis=1),color='b',linewidth=2)
ax[0].set_xlim(nino34_years[0],2000)
ax[0].set_title("(a) Annual Nino3.4 index",fontsize=18,loc='left')
ax[0].text(0.75,0.87,"$R$="+str('%1.2f' % nino34_correlation_post)+", $CE$="+str('%1.2f' % nino34_CE_post),fontsize=16,transform=ax[0].transAxes)
ax[0].set_ylabel("Nino3.4",fontsize=16)

line1, = ax[1].plot(pdo_obs_years,pdo_obs_annual,color='k',linewidth=2)
line2  = ax[1].fill_between(years_lmr,np.percentile(pdo_lmr,2.5,axis=1),np.percentile(pdo_lmr,97.5,axis=1),facecolor='b',alpha=0.2)
line3, = ax[1].plot(years_lmr,np.mean(pdo_lmr,axis=1),color='b',linewidth=2)
ax[1].set_xlim(pdo_obs_years[0],2000)
ax[1].set_title("(b) Annual Pacific Decadal Oscillation (PDO)",fontsize=18,loc='left')
ax[1].text(0.75,0.87,"$R$="+str('%1.2f' % pdo_correlation_post)+", $CE$="+str('%1.2f' % pdo_CE_post),fontsize=16,transform=ax[1].transAxes)
ax[1].set_ylabel("PDO",fontsize=16)

line1, = ax[2].plot(amo_obs_years,amo_obs_annual,color='k',linewidth=2)
line2  = ax[2].fill_between(amo_years_new,np.percentile(amo_new_lmr,2.5,axis=1),np.percentile(amo_new_lmr,97.5,axis=1),facecolor='b',alpha=0.2)
line3, = ax[2].plot(amo_years_new,np.mean(amo_new_lmr,axis=1),color='b',linewidth=2)
ax[2].set_xlim(amo_obs_years[0],2000)
ax[2].set_title("(c) Annual Atlantic Multidecadal Oscillation (AMO)",fontsize=18,loc='left')
ax[2].text(0.75,0.87,"$R$="+str('%1.2f' % amo_new_correlation_post)+", $CE$="+str('%1.2f' % amo_new_CE_post),fontsize=16,transform=ax[2].transAxes)
ax[2].set_xlabel("Year",fontsize=16)
ax[2].set_ylabel("AMO",fontsize=16)

legend = plt.legend([line1,line3],["Instrument-based","LMR"],loc=1,ncol=3,bbox_to_anchor=(1,-.08),prop={'size':14})
legend.get_frame().set_alpha(0.5)

f.suptitle("Annual-mean climate indices",fontsize=22)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot:
    plt.savefig("figures/SuppFig2_climate_indices_"+experiment_name+".png",dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


# FIGURE: SOI
plt.figure(figsize=(15,4))
plt.axes([.05,.15,.9,.75])
line1, = plt.plot(soi_obs_years,soi_obs_annual,color='k',linewidth=2)
line2 = plt.fill_between(years_lmr,np.percentile(soi_lmr,2.5,axis=1),np.percentile(soi_lmr,97.5,axis=1),facecolor='b',alpha=0.2)
line3, = plt.plot(years_lmr,np.mean(soi_lmr,axis=1),color='b',linewidth=2)
plt.xlim(soi_obs_years[0],2000)
legend = plt.legend([line1,line3],["data","LMR"],loc=1,ncol=3,bbox_to_anchor=(1,-.08),prop={'size':11})
legend.get_frame().set_alpha(0.5)
plt.title("(d) Annual Southern Oscillation Index (SOI).  $R$="+str('%1.2f' % soi_correlation_post)+", $CE$="+str('%1.2f' % soi_CE_post),fontsize=24)
plt.xlabel("Year",fontsize=16)
plt.ylabel("SOI",fontsize=16)
if save_instead_of_plot:
    plt.savefig("figures/SuppFig2_extra_soi_"+experiment_name+".png",dpi=300,format='png',bbox_inches='tight')
else:
    plt.show()

# FIGURE: AMO, old formulation
plt.figure(figsize=(15,4))
plt.axes([.05,.15,.9,.75])
line1, = plt.plot(amo_obs_years,amo_obs_annual,color='k',linewidth=2)
line2 = plt.fill_between(years_lmr,np.percentile(amo_lmr,2.5,axis=1),np.percentile(amo_lmr,97.5,axis=1),facecolor='b',alpha=0.2)
line3, = plt.plot(years_lmr,np.mean(amo_lmr,axis=1),color='b',linewidth=2)
plt.xlim(amo_obs_years[0],2000)
legend = plt.legend([line1,line3],["data","LMR"],loc=1,ncol=3,bbox_to_anchor=(1,-.08),prop={'size':11})
legend.get_frame().set_alpha(0.5)
plt.title("Annual Atlantic Multidecadal Oscillation (AMO), old formulation.  $R$="+str('%1.2f' % amo_correlation_post)+", $CE$="+str('%1.2f' % amo_CE_post),fontsize=24)
plt.xlabel("Year",fontsize=16)
plt.ylabel("Nino3.4",fontsize=16)
if save_instead_of_plot:
    plt.savefig("figures/SuppFig4_extra_old_"+experiment_name+".png",dpi=300,format='png',bbox_inches='tight')
else:
    plt.show()

