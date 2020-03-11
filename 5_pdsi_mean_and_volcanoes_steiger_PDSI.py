#==============================================================================
# This script examines PDSI characteristics in the CESM LME all-forcing
# simulations.  There are thirteen of these simulations, and the correlations
# between all pairs of simulations are examined.
#   author: Michael P. Erb
#   date  : 9/26/2018
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import compute_regional_means
import mpe_functions as mpe
import h5py

# Function to compute annual means
#var = pdsi_monthly
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

save_instead_of_plot = True
data_dir = '/projects/pd_lab/data/models/CESM_LME/'
data_dir_pdsi = '/projects/pd_lab/data/models/CESM_LME/PDSI_from_NathanSteiger/'
experiments = ['002','003','004','005','006','007','008','009','010']

region_bounds_sw           = [32,  40,-125,  -105]  # 32-40N, 125-105W
#region_bounds_southcentral = [28,  40,-115,   -95]  # 28-40N, 115-95W



### LOAD DATA

# Load LMR data
handle_basics = xr.open_dataset(data_dir+'b.e11.BLMTRC5CN.f19_g16.001.cam.h0.PDSI.085001-184912.nc',decode_times=False)
lat  = handle_basics['lat'].values
lon  = handle_basics['lon'].values
time = handle_basics['time'].values
handle_basics.close()
years = np.arange(850,1850)

nlat = len(lat)
nlon = len(lon)

# Compute days per month
time_appended = np.insert(time,0,0)
days_per_month = np.diff(time_appended)

pdsi = {}
nexperiments = len(experiments)
for experiment in experiments:
    #
    print('Loading data for: '+experiment)
    data_pdsi = h5py.File(data_dir_pdsi+'cesm_lme_'+experiment+'_pdsi_output_AWC_c_7_u2_gcm.mat','r')
    pdsi_monthly = data_pdsi['pdsi_f'][:]
    data_pdsi.close()
    #
    # Swtich dimensions in Nathan's PDSI file
    pdsi_monthly = np.swapaxes(pdsi_monthly,1,2)
    #
    pdsi[experiment+'_annual'] = annual_mean(pdsi_monthly,days_per_month)
    del pdsi_monthly

# Load the volcanic forcing data, which runs from December, 500, to January, 2001.
handle_volc = xr.open_dataset(data_dir+'forcings/IVI2LoadingLatHeight501-2000_L18_c20100518.nc',decode_times=False)
colmass_volc = handle_volc['colmass'].values
date_volc    = handle_volc['date'].values
lat_volc     = handle_volc['lat'].values
handle_volc.close()

years_volc = np.arange(501,2001)


### CALCULATIONS

# Compute mean PDSI for the southwest U.S.
for experiment in experiments:
    pdsi[experiment+'_annual_sw'] = compute_regional_means.compute_means(pdsi[experiment+'_annual'],lat,lon,region_bounds_sw[0],region_bounds_sw[1],region_bounds_sw[2],region_bounds_sw[3])
#    pdsi[experiment+'_annual_southcentral'] = compute_regional_means.compute_means(pdsi[experiment+'_annual'],lat,lon,region_bounds_southcentral[0],region_bounds_southcentral[1],region_bounds_southcentral[2],region_bounds_southcentral[3])

# Calculate the mean PDSI for all of the simulations
nyears = len(years)
pdsi_sw_allsims = np.zeros((nexperiments,nyears)); pdsi_sw_allsims[:] = np.nan
for i,experiment in enumerate(experiments):
    pdsi_sw_allsims[i,:] = pdsi[experiment+'_annual_sw']

pdsi_sw_mean = np.mean(pdsi_sw_allsims,axis=0)


### VOLCANIC CALCULATIONS

# Compute a global mean from the latitudinal data
lat_volc_weights = np.cos(np.radians(lat_volc))
colmass_volc_global = np.average(colmass_volc,axis=1,weights=lat_volc_weights)

# Remove the first and last data point, so that only data for full years is retained.
colmass_volc_fullyears = colmass_volc_global[1:-1]
nyears_volc = int(len(colmass_volc_fullyears)/12)
colmass_volc_fullyears_2d = np.reshape(colmass_volc_fullyears,(nyears_volc,12))

# Compute annual-means
colmass_volc_annual = mpe.annual_mean(years_volc,colmass_volc_fullyears_2d)

# Find the 10 largest values
nevents = 10
extreme_index = colmass_volc_annual.argsort()[-nevents*2:][::-1]
extreme_years = years_volc[extreme_index]

# Remove the later years of multiyear events
extreme_years_firstyears = []
for i in range(nevents*2):
    candidate_year = extreme_years[i]
    if candidate_year-1 in extreme_years:
        if (candidate_year-1 not in extreme_years_firstyears):
            extreme_years_firstyears.append(candidate_year-1)
        else:
            continue
    else:
        if (candidate_year not in extreme_years_firstyears):
            extreme_years_firstyears.append(candidate_year)
        else:
            continue

extreme_years_final = extreme_years_firstyears[0:nevents]



### FIGURES
plt.style.use('ggplot')

# Plot time series of southwest U.S. annual-mean PDSI in the mean of all simulations
f, ax = plt.subplots(4,1,figsize=(13,13),sharey=True)
ax = ax.ravel()

year_limits = np.array([[850,1100],[1100,1350],[1350,1600],[1600,1850]])
for panel in range(4):
    for i,experiment in enumerate(experiments):
        pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw']
        line_individual, = ax[panel].plot(years,pdsi_total_sw_anomaly,color='gray',linewidth=.75)
    #
    line_mean, = ax[panel].plot(years,pdsi_sw_mean,color='k',linewidth=2)
    ax[panel].plot(years,pdsi_sw_mean*0,'--',color='k')
    for i in range(nevents):
        ax[panel].axvline(x=extreme_years_final[i],color='r',linewidth=2)
        if (extreme_years_final[i] >= year_limits[panel,0]) & (extreme_years_final[i] < year_limits[panel,1]):
            if extreme_years_final[i] == 1809: ax[panel].text(extreme_years_final[i]-.5,7.2,str(extreme_years_final[i]),horizontalalignment='right',verticalalignment='center',color='r')
            else:                              ax[panel].text(extreme_years_final[i]+.5,7.2,str(extreme_years_final[i]),horizontalalignment='left', verticalalignment='center',color='r')
    #
    ax[panel].set_xlim(year_limits[panel,:])
    ax[panel].set_ylim(-8,8)
    ax[panel].set_ylabel('PDSI',fontsize=16)
    ax[panel].tick_params(labelsize=12)
    if panel == 0: ax[panel].legend([line_mean,line_individual],['Mean','Individual simulations'])
    if panel == 3: ax[panel].set_xlabel('Year C.E.',fontsize=16)

plt.suptitle('Annual-mean southwest U.S. PDSI, all-forcing simulations',fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=.95)
if save_instead_of_plot == True:
    plt.savefig('figures/5_allforcing_pdsi_sw_ts_mean_and_volcanoes_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



# Plot time series of southwest U.S. annual-mean PDSI near the Samalas eruption of 1257
plt.figure(figsize=(10,8))
ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
ax2 = plt.subplot2grid((3,1),(2,0))

for i,experiment in enumerate(experiments):
    pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw']
    line_individual, = ax1.plot(years,pdsi_total_sw_anomaly,color='gray',linewidth=.75)

line_mean, = ax1.plot(years,pdsi_sw_mean,color='k',linewidth=2)
ax1.plot(years,pdsi_sw_mean*0,'--',color='k')
ax1.axvline(x=1257,color='r',linewidth=2)
ax1.text(1257.2,6.5,'1257',horizontalalignment='left',verticalalignment='center',color='r')

ax1.set_xlim(1250,1270)
ax1.set_ylim(-7,7)
ax1.set_xticks(np.arange(1250,1271,2))
ax1.set_ylabel('PDSI',fontsize=16)
ax1.tick_params(labelsize=12)
ax1.legend([line_mean,line_individual],['Mean','Individual simulations'])
ax1.set_xlabel('Year',fontsize=16)
ax1.set_title('(a) Annual-mean southwest U.S. PDSI',loc='left',fontsize=16)

ax2.plot(years_volc,colmass_volc_annual,color='k',linewidth=1)
ax2.set_xlim(1250,1270)
ax2.set_xticks(np.arange(1250,1271,2))
ax2.set_xlabel('Year',fontsize=16)
ax2.set_ylabel('Aerosols (kg m$^{-2}$)',fontsize=16)  # Right?
ax2.set_ylim(ymin=0)
ax2.set_title('(b) Mean volcanic column aerosol mass',loc='left',fontsize=16)

plt.suptitle('Response to the Samalas eruption of 1257',fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=.9)
if save_instead_of_plot == True:
    plt.savefig('figures/allforcing_pdsi_sw_ts_mean_and_Samalas_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()

