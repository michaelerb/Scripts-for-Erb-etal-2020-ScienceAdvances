#==============================================================================
# This script compares different self-organizing maps during drought years vs.
# non-drought years.
#   author: Michael P. Erb
#   date  : 11/22/2019
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
from scipy import stats
import numpy.ma as ma
import sys


save_instead_of_plot = False

# Choose the SOM analysis to use
#n_patterns = int(sys.argv[1])
n_patterns = 8
som_analysis = 'soms_sst';               som_var = 'tmp';  title_txt = 'normal';           som_filename_txt1 = 'soms_sst_output_lmr_som';     som_filename_txt2 = '_yrs10011925_22-Nov-2019.nc'
#som_analysis = 'soms_sst_noENSO';        som_var = 'tmp';  title_txt = 'no ENSO';          som_filename_txt1 = 'soms_sst_output_lmr_som';     som_filename_txt2 = '_yrs10011925_02-Dec-2019_noENSO.nc'
#som_analysis = 'soms_z500';              som_var = 'z500'; title_txt = 'z500 normal';      som_filename_txt1 = 'soms_z500_output_lmr_som';    som_filename_txt2 = '_yrs10011925_02-Dec-2019.nc'
#som_analysis = 'soms_z500_standardized'; som_var = 'z500';title_txt = 'z500 standardized'; som_filename_txt1 = 'soms_z500std_output_lmr_som'; som_filename_txt2 = '_yrs10011925_02-Dec-2019.nc'

calc_bounds = [-25,70,90,360]  # Specify the region to plot over
data_dir     = '/projects/pd_lab/data/LMR/archive_output/'
revision_dir = '/home/mpe32/analysis/5_drought/revisions_paper_v2/'

subplot_size = {}
subplot_size[6]  = [3,2]
subplot_size[8]  = [4,2]
subplot_size[12] = [4,3]
subplot_size[16] = [4,4]
subplot_size[20] = [5,4]


### LOAD DATA

# Load LMR output from the production run
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


# Load the self-organizing maps
som_patterns = {}
handle = xr.open_dataset(revision_dir+'self_organizing_maps/data/'+som_analysis+'/'+som_filename_txt1+str(n_patterns)+som_filename_txt2,decode_times=False)
for i in range(1,n_patterns+1):
    som_patterns[i] = handle[som_var+'_ptrn_'+str(i)].values

pattern_of_year  = handle['bmus'].values
years_som        = handle['time'].values
handle.close()

# Load the list of drought and non-drought years
drought_year_data = np.load(revision_dir+'data/drought_years.npz')
drought_years = {}
drought_years['northwest'] = drought_year_data['drought_years_northwest']
drought_years['southwest'] = drought_year_data['drought_years_southwest']
drought_years['central']   = drought_year_data['drought_years_central']
drought_years['southeast'] = drought_year_data['drought_years_southeast']


### CALCULATIONS

# Shorten the data to cover only the desired years.
year_bounds = [1001,1925]
indices_chosen = np.where((years_all >= year_bounds[0]) & (years_all <= year_bounds[1]))[0]
scpdsi_chosen    = scpdsi_all[indices_chosen,:,:,:]
sst_chosen       = sst_all[indices_chosen,:,:,:]
zg_500hPa_chosen = zg_500hPa_all[indices_chosen,:,:,:]
years_chosen     = years_all[indices_chosen]

# Compute a mean of all iterations
scpdsi_mean    = np.mean(scpdsi_chosen,   axis=1)
sst_mean       = np.mean(sst_chosen,      axis=1)
zg_500hPa_mean = np.mean(zg_500hPa_chosen,axis=1)

# Mask the variables
scpdsi_mean = scpdsi_mean*oceanmask[None,:,:]
scpdsi_mean = ma.masked_invalid(scpdsi_mean)

# A function to detrend quantities at every gridpoint
#quantity_mean = sst_mean
def detrend_spatial(quantity_mean):
    #
    global years_chosen
    quantity_detrended = np.zeros((quantity_mean.shape)); quantity_detrended[:] = np.nan
    nlat = quantity_mean.shape[1]
    nlon = quantity_mean.shape[2]
    #
    for j in range(nlat):
        for i in range(nlon):
            quantity_selected = quantity_mean[:,j,i]
            slope,intercept,r_value,p_value,std_err = stats.linregress(years_chosen,quantity_selected)
            quantity_selected_linear = slope*years_chosen + intercept
            quantity_selected_detrended = quantity_selected - quantity_selected_linear
            #
            quantity_detrended[:,j,i] = quantity_selected_detrended
    #
    return quantity_detrended

# Detrend quantities at every gridpoint
scpdsi_mean    = detrend_spatial(scpdsi_mean)
sst_mean       = detrend_spatial(sst_mean)
zg_500hPa_mean = detrend_spatial(zg_500hPa_mean)

# For each pattern, compute the mean climate fields
scpdsi_pattern_lmr = {}; sst_pattern_lmr = {}; zg_500hPa_pattern_lmr = {}
for i in range(n_patterns):
    years_of_pattern = years_som[pattern_of_year == i+1]
    lmr_indicies_selected = np.nonzero(np.in1d(years_chosen,years_of_pattern))[0]
    scpdsi_pattern_lmr[i+1]    = np.mean(scpdsi_mean[lmr_indicies_selected,:,:],   axis=0)
    sst_pattern_lmr[i+1]       = np.mean(sst_mean[lmr_indicies_selected,:,:],      axis=0)
    zg_500hPa_pattern_lmr[i+1] = np.mean(zg_500hPa_mean[lmr_indicies_selected,:,:],axis=0)

# Calculate the number of times that each pattern appears in all years
pattern_num        = np.zeros(n_patterns); pattern_num[:]        = np.nan
pattern_counts_all = np.zeros(n_patterns); pattern_counts_all[:] = np.nan
for i in range(n_patterns):
    pattern_num[i] = i+1
    pattern_counts_all[i] = sum(pattern_of_year == i+1)

# Calculate the relative number of times that each pattern appears
percent_pattern_all = 100*(pattern_counts_all / sum(pattern_counts_all))

# Find the indices of drought years and non-drought years
regions = drought_years.keys()
indices_drought = {}; indices_nondrought = {}
for region in regions:
    indices_drought[region]    = np.nonzero(np.in1d(years_som,drought_years[region]))[0]
    indices_nondrought[region] = np.nonzero(~np.in1d(years_som,drought_years[region]))[0]

# Calculate the number of times that each pattern appears in drought years and non-drought years for different regions
pattern_counts_drought  = {}; pattern_counts_nondrought  = {}
percent_pattern_drought = {}; percent_pattern_nondrought = {}
for region in regions:
    pattern_counts_drought[region]    = np.zeros(n_patterns); pattern_counts_drought[region][:]    = np.nan
    pattern_counts_nondrought[region] = np.zeros(n_patterns); pattern_counts_nondrought[region][:] = np.nan
    for i in range(n_patterns):
        pattern_counts_drought[region][i]    = sum(pattern_of_year[indices_drought[region]]    == i+1)
        pattern_counts_nondrought[region][i] = sum(pattern_of_year[indices_nondrought[region]] == i+1)
    #
    # Calculate the relative number of times that each pattern appears
    percent_pattern_drought[region]    = 100*(pattern_counts_drought[region]    / sum(pattern_counts_drought[region]))
    percent_pattern_nondrought[region] = 100*(pattern_counts_nondrought[region] / sum(pattern_counts_nondrought[region]))



### FIGURES

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']

# Map
m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c')
lon_2d,lat_2d = np.meshgrid(lon,lat)
x, y = m(lon_2d,lat_2d)

f, ax = plt.subplots(subplot_size[n_patterns][0],subplot_size[n_patterns][1],figsize=(15,12),sharex=True,sharey=True)
ax = ax.ravel()

for i in range(n_patterns):
    #
    pattern_selected = som_patterns[i+1]
    #
    m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c',ax=ax[i])
    if som_var == 'tmp':
        image1 = m.contourf(x,y,pattern_selected,np.linspace(-1.6,1.6,17),extend='both',cmap='RdBu_r',vmin=-1.6,vmax=1.6)
        image3 = m.contour( x,y,zg_500hPa_pattern_lmr[i+1],np.linspace(-30,30,31),colors='k',linewidths=1)
    elif som_var == 'z500':
        image1 = m.contourf(x,y,sst_pattern_lmr[i+1],np.linspace(-.5,.5,11),extend='both',cmap='RdBu_r',vmin=-.5,vmax=.5)
        if   som_analysis == 'soms_z500':              contour_values = np.linspace(-30,30,31)
        elif som_analysis == 'soms_z500_standardized': contour_values = np.linspace(-3,3,31)
        image3 = m.contour( x,y,pattern_selected,contour_values,colors='k',linewidths=1)
    #
    image2 = m.contourf(x,y,scpdsi_pattern_lmr[i+1],   np.linspace(-1,1,11),extend='both',cmap='BrBG',vmin=-1,vmax=1)
    m.drawparallels([0],labels=[True])
    m.drawcoastlines()
    m.colorbar(image1)
    m.colorbar(image2,location='bottom')
    ax[i].set_title(letters[i]+') Pattern '+str(i+1)+' (n='+str(int(pattern_counts_all[i]))+')',loc='left',fontsize=20)

f.suptitle('SST patterns of the self-organizing maps ('+title_txt+'), $N_{patterns}$='+str(n_patterns),fontsize=24)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot == True:
    plt.savefig('figures/som_'+som_analysis+'_maps_npatterns_'+str(n_patterns).zfill(2)+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


# Make a bar plot of the relative proportion of drought and non-drought years which correspond to each SOM
f = plt.figure(figsize=(17,5))
region_to_plot = 'southwest'
bar_width = 0.4
plt.bar(pattern_num-bar_width/2, percent_pattern_nondrought[region_to_plot], bar_width, label='Non-drought years')
plt.bar(pattern_num+bar_width/2, percent_pattern_drought[region_to_plot],    bar_width, label='Drought years')
plt.title('Percentage of drought and non-drought years for the '+region_to_plot+' U.S.\ncorresponding to each map ('+title_txt+'), $N_{patterns}$='+str(n_patterns),fontsize=18)
plt.xlabel('Map number',fontsize=18)
plt.ylabel('Percentage (%)',fontsize=18)
plt.legend(fontsize=14)
if save_instead_of_plot == True:
    plt.savefig('figures/som_'+som_analysis+'_percentage_'+region_to_plot+'_npatterns_'+str(n_patterns).zfill(2)+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


# Make a bar plot of the relative proportion of drought years which correspond to each SOM for each region
f = plt.figure(figsize=(17,5))
bar_width = 0.15
plt.bar(pattern_num-(bar_width*2), percent_pattern_all,bar_width, label='All years', color='k')
for i,region in enumerate(regions):
    plt.bar(pattern_num-((1-i)*bar_width), percent_pattern_drought[region], bar_width, label='Drought years in '+region+' US')

plt.title('Percentage of years corresponding to each map ('+title_txt+'), $N_{patterns}$='+str(n_patterns),fontsize=18)
plt.xlabel('Map number',fontsize=18)
plt.ylabel('Percentage (%)',fontsize=18)
plt.legend(fontsize=14)
if save_instead_of_plot == True:
    plt.savefig('figures/som_'+som_analysis+'_percentage_all_regions_npatterns_'+str(n_patterns).zfill(2)+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


# Make a bar plot to show the changes in relative proportion of drought years which correspond to each SOM for each region
f = plt.figure(figsize=(17,5))
bar_width = 0.15
for i,region in enumerate(regions):
    plt.bar(pattern_num-((1.5-i)*bar_width), percent_pattern_drought[region]-percent_pattern_all, bar_width, label=region.capitalize()+' US')

plt.axhline(y=0,c='k',linewidth=1,linestyle='--')
plt.title('i) Change in percentage of years corresponding to each pattern ('+title_txt+')',loc='left',fontsize=20)
plt.xlabel('Map number',fontsize=18)
plt.ylabel('$\Delta$ percentage (%)',fontsize=18)
plt.legend(fontsize=14)
if save_instead_of_plot == True:
    plt.savefig('figures/som_'+som_analysis+'_percentage_all_regions_change_npatterns_'+str(n_patterns).zfill(2)+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()



# Figure for paper
m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c')
lon_2d,lat_2d = np.meshgrid(lon,lat)
x, y = m(lon_2d,lat_2d)

f = plt.figure(figsize=(15,15))
ax = {}
ax[0]  = plt.subplot2grid((5,2),(0,0))
ax[1]  = plt.subplot2grid((5,2),(0,1))
ax[2]  = plt.subplot2grid((5,2),(1,0))
ax[3]  = plt.subplot2grid((5,2),(1,1))
ax[4]  = plt.subplot2grid((5,2),(2,0))
ax[5]  = plt.subplot2grid((5,2),(2,1))
ax[6]  = plt.subplot2grid((5,2),(3,0))
ax[7]  = plt.subplot2grid((5,2),(3,1))
ax[8]  = plt.subplot2grid((5,2),(4,0),colspan=2)

for i in range(n_patterns):
    #
    pattern_selected = som_patterns[i+1]
    #
    m = Basemap(projection='cyl',llcrnrlat=calc_bounds[0],urcrnrlat=calc_bounds[1],llcrnrlon=calc_bounds[2],urcrnrlon=calc_bounds[3],resolution='c',ax=ax[i])
    if som_var == 'tmp':
        image1 = m.contourf(x,y,pattern_selected,np.linspace(-1.6,1.6,17),extend='both',cmap='RdBu_r',vmin=-1.6,vmax=1.6)
        image3 = m.contour( x,y,zg_500hPa_pattern_lmr[i+1],np.linspace(-30,30,31),colors='k',linewidths=1)
    elif som_var == 'z500':
        image1 = m.contourf(x,y,sst_pattern_lmr[i+1],np.linspace(-.5,.5,11),extend='both',cmap='RdBu_r',vmin=-.5,vmax=.5)
        if   som_analysis == 'soms_z500':              contour_values = np.linspace(-30,30,31)
        elif som_analysis == 'soms_z500_standardized': contour_values = np.linspace(-3,3,31)
        image3 = m.contour( x,y,pattern_selected,contour_values,colors='k',linewidths=1)
    #
    image2 = m.contourf(x,y,scpdsi_pattern_lmr[i+1],   np.linspace(-1,1,11),extend='both',cmap='BrBG',vmin=-1,vmax=1)
    m.drawparallels([0],labels=[True])
    m.drawcoastlines()
    m.colorbar(image1)
    m.colorbar(image2,location='bottom')
    ax[i].set_title(letters[i]+') Pattern '+str(i+1)+' (n='+str(int(pattern_counts_all[i]))+')',loc='left',fontsize=20)

# Make a bar plot to show the changes in relative proportion of drought years which correspond to each SOM for each region
bar_width = 0.15
for i,region in enumerate(regions):
    ax[8].bar(pattern_num-((1.5-i)*bar_width), percent_pattern_drought[region]-percent_pattern_all, bar_width, label=region.capitalize()+' US')

ax[8].axhline(y=0,c='k',linewidth=1,linestyle='--')
ax[8].set_title('i) Change in percentage of years corresponding to each pattern',loc='left',fontsize=20)
ax[8].set_xlabel('Pattern number',fontsize=18)
ax[8].set_ylabel('$\Delta$ percentage (%)',fontsize=18)
ax[8].legend(fontsize=14,ncol=2)

f.suptitle('SST patterns of the self-organizing maps',fontsize=24)
f.tight_layout()
f.subplots_adjust(top=.92)
if save_instead_of_plot == True:
    plt.savefig('figures/FigS7_som_'+som_analysis+'_maps_npatterns_'+str(n_patterns).zfill(2)+'.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()

