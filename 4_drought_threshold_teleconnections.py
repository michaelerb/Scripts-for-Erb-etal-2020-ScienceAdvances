#=============================================================================
# This script looks at the mean climate patterns during drought events.
#    author: Michael P. Erb
#    date  : 12/12/2019
#=============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import xarray as xr
import compute_regional_means
from scipy import stats
import seaborn as sns
import pandas as pd


save_instead_of_plot = True
drought_criteria_txt = 'standard_dev_timevarying'
#drought_criteria_txt = 'standard_dev'
#drought_criteria_txt = '90th_percentile'
#drought_criteria_txt = '95th_percentile'


### LOAD DATA

# Load data from the production run
data_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name1 = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
experiment_name2 = experiment_name1

handle = xr.open_dataset(data_dir+experiment_name1+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_all = handle['pdsi'].values
lon        = handle['lon'].values
lat        = handle['lat'].values
time       = handle['time'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2+'/sst_MCruns_ensemble_mean.nc',decode_times=False)
sst_all = handle['sst'].values
handle.close()

handle = xr.open_dataset(data_dir+experiment_name2+'/hgt500_MCruns_ensemble_mean.nc',decode_times=False)
zg_500hPa_all = handle['hgt500'].values
handle.close()

years_all = time/365
years_all = years_all.astype(int)

# Open the landmask file
handle = xr.open_dataset('/home/mpe32/analysis/5_drought/masks/output/oceanmask_landmask_lmr.nc',decode_times=False)
oceanmask = handle['oceanmask'].values
handle.close()

# Load the calculated climate indices
handle = xr.open_dataset(data_dir+experiment_name1+'/climate_indices_MCruns_ensemble_mean_calc_from_posterior.nc',decode_times=False)
nino34_all = handle['nino34'].values
soi_all    = handle['soi'].values
amo_all    = handle['amo'].values
pdo_all    = handle['pdo'].values
handle.close()


### CALCULATIONS

# Shorten the data to cover only the desired years.
year_bounds = [1001,2000]
indices_chosen = np.where((years_all >= year_bounds[0]) & (years_all <= year_bounds[1]))[0]
scpdsi_all    = scpdsi_all[indices_chosen,:,:,:]
sst_all       = sst_all[indices_chosen,:,:,:]
zg_500hPa_all = zg_500hPa_all[indices_chosen,:,:,:]
nino34_all    = nino34_all[indices_chosen,:]
soi_all       = soi_all[indices_chosen,:]
amo_all       = amo_all[indices_chosen,:]
pdo_all       = pdo_all[indices_chosen,:]
years         = years_all[indices_chosen]

# Remove the mean values for each quantity.
scpdsi_all    = scpdsi_all    - np.mean(scpdsi_all,   axis=0)[None,:,:,:]
sst_all       = sst_all       - np.mean(sst_all,      axis=0)[None,:,:,:]
zg_500hPa_all = zg_500hPa_all - np.mean(zg_500hPa_all,axis=0)[None,:,:,:]
nino34_all    = nino34_all    - np.mean(nino34_all,   axis=0)[None,:]
soi_all       = soi_all       - np.mean(soi_all,      axis=0)[None,:]
amo_all       = amo_all       - np.mean(amo_all,      axis=0)[None,:]
pdo_all       = pdo_all       - np.mean(pdo_all,      axis=0)[None,:]

# Compute a mean of all iterations
scpdsi_mean    = np.mean(scpdsi_all,   axis=1)
sst_mean       = np.mean(sst_all,      axis=1)
zg_500hPa_mean = np.mean(zg_500hPa_all,axis=1)
nino34_mean    = np.mean(nino34_all,   axis=1)
soi_mean       = np.mean(soi_all,      axis=1)
amo_mean       = np.mean(amo_all,      axis=1)
pdo_mean       = np.mean(pdo_all,      axis=1)
#plt.contourf(np.mean(scpdsi_mean,axis=0)); plt.colorbar()

# Detrend everything

# Mask the variables
scpdsi_mean = scpdsi_mean*oceanmask[None,:,:]
scpdsi_mean = ma.masked_invalid(scpdsi_mean)

# Compute average of PDSI from atlas and LMR for the entire region as well as the four regions used in Cook et al. 2014.
pdsi_mean_regions = compute_regional_means.compute_US_means(scpdsi_mean,lat,lon)

# Compute means over the drought indices
#pdsi_region_selected = pdsi_mean_regions['southwest']
def drought_means(pdsi_region_selected):
    #
    global drought_criteria_txt,scpdsi_mean,sst_mean,zg_500hPa_mean
    #
    # Select the right indices for drought
    if drought_criteria_txt == 'standard_dev_timevarying':
        nyears = len(pdsi_region_selected)
        segment_length_oneside = 25
        drought_indices    = []
        nondrought_indices = []
        pluvial_indices    = []
        for i in range(nyears):
            seg_start = i-segment_length_oneside
            seg_end   = i+segment_length_oneside
            if seg_start < 0:        seg_start = 0;         seg_end = 50
            if seg_end   > nyears-1: seg_start = nyears-51; seg_end = nyears-1
            print(i,seg_start,seg_end)
            pdsi_selected            = pdsi_region_selected[i]                   - np.mean(pdsi_region_selected[seg_start:seg_end+1])
            pdsi_region_selected_seg = pdsi_region_selected[seg_start:seg_end+1] - np.mean(pdsi_region_selected[seg_start:seg_end+1])
#            pdsi_region_selected_seg = pdsi_region_selected[seg_start:seg_end+1]
            drought_criteria = -1*np.std(pdsi_region_selected_seg)
            #
            if pdsi_selected <  drought_criteria:   drought_indices.append(i)
            if pdsi_selected >= drought_criteria:   nondrought_indices.append(i)
            if pdsi_selected > -1*drought_criteria: pluvial_indices.append(i)
        #
        drought_indices    = np.array(drought_indices)
        nondrought_indices = np.array(nondrought_indices)
        pluvial_indices    = np.array(pluvial_indices)
        #
    else:
        if   drought_criteria_txt == 'standard_dev':    drought_criteria = -1*np.std(pdsi_region_selected)
        elif drought_criteria_txt == '90th_percentile': drought_criteria = np.percentile(pdsi_region_selected,10)
        elif drought_criteria_txt == '95th_percentile': drought_criteria = np.percentile(pdsi_region_selected,5)
        #
        # Select the right indices
        drought_indices    = np.where(pdsi_region_selected <  drought_criteria)[0]
        nondrought_indices = np.where(pdsi_region_selected >= drought_criteria)[0]
        pluvial_indices    = np.where(pdsi_region_selected > -1*drought_criteria)[0]
    #
    # Compute means over the drought indices
    scpdsi_in_drought    = np.mean(scpdsi_mean[drought_indices,:,:],   axis=0)
    sst_in_drought       = np.mean(sst_mean[drought_indices,:,:],      axis=0)
    zg_500hPa_in_drought = np.mean(zg_500hPa_mean[drought_indices,:,:],axis=0)
    #
    return drought_indices,nondrought_indices,pluvial_indices,scpdsi_in_drought,sst_in_drought,zg_500hPa_in_drought

# Find the indeices which are in drought at every time point
regions = pdsi_mean_regions.keys()
drought_indices_all = {}; nondrought_indices_all = {}; scpdsi_in_drought_all = {}; sst_in_drought_all = {}; zg_500hPa_in_drought_all = {}
for region in regions:
    drought_indices_all[region],nondrought_indices_all[region],_,scpdsi_in_drought_all[region],sst_in_drought_all[region],zg_500hPa_in_drought_all[region] = drought_means(pdsi_mean_regions[region])

# Calculate the percentage of time that drought years have a negative Nino3.4 index, for each region
percent_LaNina_allyears = (sum(nino34_mean < 0) / len(nino34_mean))*100
print('==============================================')
print('Percentage of years with below-average Nino3.4')
print('==============================================')
print('All years: '+str(percent_LaNina_allyears))
for region in ['northwest','southwest','central','southeast']:
    nino34_selected = nino34_mean[drought_indices_all[region]]
    percent_LaNina_selected = (sum(nino34_selected < np.mean(nino34_mean)) / len(nino34_selected))*100
    print('Drought years in '+region+' US: '+str(percent_LaNina_selected))



### FIGURES
plt.style.use('ggplot')

# Make a time series of regional PDSI and "drought" years
for region in ['northwest','southwest','central','southeast']:
    #
    f = plt.figure(figsize=(16,4))
    print(region,drought_indices_all[region].shape)
    for drought_index in drought_indices_all[region]:
        plt.axvline(x=years[drought_index],c='orangered')
    #
    plt.plot(years,pdsi_mean_regions[region],c='k')
    plt.xlim(year_bounds)
    plt.xlabel('Year (C.E.)')
    plt.ylabel('PDSI')
    plt.title('Regional drought for the '+region+' U.S., with drought years marked',fontsize=24)
    if save_instead_of_plot == True:
        plt.savefig('figures/pdsi_ts_'+region+'_'+drought_criteria_txt+'.png',dpi=300,format='png')
        plt.close()
    else:
        plt.show()



# Specify the region to plot over
calc_bounds = [-25,70,90,360]

# Map
m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c')
lon_2d,lat_2d = np.meshgrid(lon,lat)
x, y = m(lon_2d,lat_2d)

# Plot the correlation between the region of interest and the selected variable everywhere.
f = plt.figure(figsize=(15,12))
ax = {}
ax[0]  = plt.subplot2grid((4,6),(0,0),colspan=3)
ax[4]  = plt.subplot2grid((4,6),(1,0),colspan=3)
ax[8]  = plt.subplot2grid((4,6),(2,0),colspan=3)
ax[12] = plt.subplot2grid((4,6),(3,0),colspan=3)
ax[1]  = plt.subplot2grid((4,6),(0,3)); ax[2]  = plt.subplot2grid((4,6),(0,4)); ax[3]  = plt.subplot2grid((4,6),(0,5))
ax[5]  = plt.subplot2grid((4,6),(1,3)); ax[6]  = plt.subplot2grid((4,6),(1,4)); ax[7]  = plt.subplot2grid((4,6),(1,5))
ax[9]  = plt.subplot2grid((4,6),(2,3)); ax[10] = plt.subplot2grid((4,6),(2,4)); ax[11] = plt.subplot2grid((4,6),(2,5))
ax[13] = plt.subplot2grid((4,6),(3,3)); ax[14] = plt.subplot2grid((4,6),(3,4)); ax[15] = plt.subplot2grid((4,6),(3,5))

regions_to_plot        = ['northwest','southwest','central','southeast']
regions_to_plot_titles = ['a) Northwest','b) Southwest','c) Central','d) Southeast']

for i,region in enumerate(regions_to_plot):
    #
    ax[0+(4*i)].set_title(regions_to_plot_titles[i]+' U.S. drought, mean conditions',fontsize=20,loc='left')
    m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c',ax=ax[0+(4*i)])
    image1 = m.contourf(x,y,scpdsi_in_drought_all[region],   np.linspace(-2.5,2.5,11),extend='both',cmap='BrBG',  vmin=-2.5,vmax=2.5)
    image2 = m.contourf(x,y,sst_in_drought_all[region],      np.linspace(-.25,.25,11),extend='both',cmap='RdBu_r',vmin=-.25,vmax=.25)
    image3 = m.contour( x,y,zg_500hPa_in_drought_all[region],np.linspace(-20,20,21),colors='k',linewidths=1)
    m.drawparallels([0],labels=[True],fontsize=12)
    m.drawcoastlines()
    cbar1 = m.colorbar(image1,location='bottom')
    cbar2 = m.colorbar(image2)
    cbar1.ax.tick_params(labelsize=12)
    cbar2.ax.tick_params(labelsize=12)
    #
    # Do some t-tests
    stat_nino34,pvalue_nino34 = stats.ttest_ind(nino34_mean[drought_indices_all[region]],nino34_mean[nondrought_indices_all[region]],axis=0,equal_var=False,nan_policy='propagate')
    stat_pdo,   pvalue_pdo    = stats.ttest_ind(pdo_mean[drought_indices_all[region]],   pdo_mean[nondrought_indices_all[region]],   axis=0,equal_var=False,nan_policy='propagate')
    stat_amo,   pvalue_amo    = stats.ttest_ind(amo_mean[drought_indices_all[region]],   amo_mean[nondrought_indices_all[region]],   axis=0,equal_var=False,nan_policy='propagate')
    note_nino34 = ''; note_pdo = ''; note_amo = ''
    if pvalue_nino34 >= 0.05: note_nino34 = '*'
    if pvalue_pdo    >= 0.05: note_pdo    = '*'
    if pvalue_amo    >= 0.05: note_amo    = '*'
    print(pvalue_nino34,pvalue_pdo,pvalue_amo)
    #
    n_drought    = len(drought_indices_all[region])
    n_nondrought = len(nondrought_indices_all[region])
    df_nino34_drought    = pd.DataFrame({'Index':['Nino34']*n_drought,   'Drought':['D']*n_drought,    'Value':nino34_mean[drought_indices_all[region]]})
    df_nino34_nondrought = pd.DataFrame({'Index':['Nino34']*n_nondrought,'Drought':['ND']*n_nondrought,'Value':nino34_mean[nondrought_indices_all[region]]})
    df_pdo_drought       = pd.DataFrame({'Index':['PDO']*n_drought,      'Drought':['D']*n_drought,    'Value':pdo_mean[drought_indices_all[region]]})
    df_pdo_nondrought    = pd.DataFrame({'Index':['PDO']*n_nondrought,   'Drought':['ND']*n_nondrought,'Value':pdo_mean[nondrought_indices_all[region]]})
    df_amo_drought       = pd.DataFrame({'Index':['AMO']*n_drought,      'Drought':['D']*n_drought,    'Value':amo_mean[drought_indices_all[region]]})
    df_amo_nondrought    = pd.DataFrame({'Index':['AMO']*n_nondrought,   'Drought':['ND']*n_nondrought,'Value':amo_mean[nondrought_indices_all[region]]})
    df_nino34 = pd.concat([df_nino34_drought,df_nino34_nondrought])
    df_pdo    = pd.concat([df_pdo_drought,df_pdo_nondrought])
    df_amo    = pd.concat([df_amo_drought,df_amo_nondrought])
    #
    sns.violinplot(x='Index',y='Value',hue='Drought',split=True,inner='quart',palette={'D':'tab:brown','ND':'tab:green'},data=df_nino34,ax=ax[1+(4*i)])
    sns.violinplot(x='Index',y='Value',hue='Drought',split=True,inner='quart',palette={'D':'tab:brown','ND':'tab:green'},data=df_pdo,   ax=ax[2+(4*i)])
    sns.violinplot(x='Index',y='Value',hue='Drought',split=True,inner='quart',palette={'D':'tab:brown','ND':'tab:green'},data=df_amo,   ax=ax[3+(4*i)])
    ax[1+(4*i)].set_title('Nino3.4'+note_nino34,fontsize=20)
    ax[2+(4*i)].set_title('PDO'+note_pdo,fontsize=20)
    ax[3+(4*i)].set_title('AMO'+note_amo,fontsize=20)
    ax[1+(4*i)].set_ylim(-2,2)
    ax[2+(4*i)].set_ylim(-4,4)
    ax[3+(4*i)].set_ylim(-.25,.25)
    ax[1+(4*i)].tick_params(axis='both',which='major',labelsize=16)
    ax[2+(4*i)].tick_params(axis='both',which='major',labelsize=16)
    ax[3+(4*i)].tick_params(axis='both',which='major',labelsize=16)
    #
    for j in range(1,4):
        if (i == 0) & (j == 1):
#            ax[j+(4*i)].legend(loc=4)
            ax[j+(4*i)].legend(loc=4,bbox_to_anchor=(1.1,0))
        else:
            ax[j+(4*i)].get_legend().remove()
        ax[j+(4*i)].set_xlabel('')
        ax[j+(4*i)].set_ylabel('')
        ax[j+(4*i)].set_xticklabels([''])

f.suptitle('Climate conditions during regional drought',fontsize=24)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot == True:
    plt.savefig('figures/climate_in_regional_drought_'+drought_criteria_txt+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


"""
# Save the drought years
output_dir = '/home/mpe32/analysis/5_drought/revisions_paper_v2/data/'
drought_years_northwest = years[drought_indices_all['northwest']]
drought_years_southwest = years[drought_indices_all['southwest']]
drought_years_central   = years[drought_indices_all['central']]
drought_years_southeast = years[drought_indices_all['southeast']]
np.savez(output_dir+'drought_years.npz',drought_years_northwest=drought_years_northwest,drought_years_southwest=drought_years_southwest,drought_years_central=drought_years_central,drought_years_southeast=drought_years_southeast)
"""
