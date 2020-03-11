#==================================================================================
# This script compares PDSI in LMR and the North American Drought Atlas (NADA).
# After regridding the NADA data to the LMR grid, correlations are computed and
# plotted.
#   author: Michael P. Erb
#   date  : 5/11/2018
#==================================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
sys.path.append('/home/mpe32/LMR/LMR/diagnostics')
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
#import pyresample
import compute_regional_means
import map_proxies_improved as map_proxies
import copy
from scipy.stats.stats import pearsonr
import mpe_functions as mpe
from matplotlib.patches import Polygon


save_instead_of_plot = True
data_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'

# Specify atlas
atlas_name = 'NADA'
#atlas_name = 'OWDA'
#atlas_name = 'MADA'
#atlas_name = 'DaiPDSI'

# Specify years
years_min = 1001
years_max = 2000
#years_min = 1
#years_max = 2000
if atlas_name == 'DaiPDSI':
    years_min = 1850
    years_max = 2000

### LOAD DATA
if atlas_name == 'NADA':
    handle_atlas = xr.open_dataset('/projects/pd_lab/data/drought_atlases/NorthAmericanDroughtAtlas/LBDA2010/nada_hd2_cl.nc',decode_times=False)
    pdsi_atlas = handle_atlas['pdsi'].values
    lat_atlas = handle_atlas['lat'].values
    lon_atlas = handle_atlas['lon'].values
    years_atlas = handle_atlas['time'].values
    lon_atlas = lon_atlas+360
    handle_atlas.close()
elif atlas_name == 'OWDA':
    handle_atlas = xr.open_dataset('/projects/pd_lab/data/drought_atlases/OldWorldDroughtAtlas/owda.nc',decode_times=False)
    pdsi_atlas = handle_atlas['pdsi'].values
    lat_atlas = handle_atlas['lat'].values
    lon_atlas = handle_atlas['lon'].values
    years_atlas = handle_atlas['time'].values
    handle_atlas.close()
elif atlas_name == 'MADA':
    handle_atlas = xr.open_dataset('/projects/pd_lab/data/drought_atlases/MonsoonAsiaDroughtAtlas/data.nc',decode_times=False)
    pdsi_atlas = handle_atlas['pdsi'].values
    lat_atlas = handle_atlas['Y'].values
    lon_atlas = handle_atlas['X'].values
    years_atlas = np.arange(1300,2006)
    handle_atlas.close()
elif atlas_name == 'DaiPDSI':
    handle_atlas = xr.open_dataset('/projects/pd_lab/data/LMR/data/analyses/DaiPDSI/Dai_pdsi.mon.mean.selfcalibrated_185001-201412.nc',decode_times=False)
    pdsi_atlas_monthly = handle_atlas['pdsi'].values
    lat_atlas          = handle_atlas['lat'].values
    lon_atlas          = handle_atlas['lon'].values
    handle_atlas.close()
    years_atlas = np.arange(1850,2015)


# Load LMR data
handle_allmeans = xr.open_dataset(data_dir+experiment_name+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_lmr_allmeans = handle_allmeans['pdsi'].values
lon_lmr             = handle_allmeans['lon'].values
lat_lmr             = handle_allmeans['lat'].values
time_lmr            = handle_allmeans['time'].values
handle_allmeans.close()

#scpdsi_lmr_subsample = np.expand_dims(scpdsi_lmr_allmeans,axis=4)
handle_subsample = xr.open_dataset(data_dir+experiment_name+'/pdsi_MCruns_ensemble_subsample.nc',decode_times=False)
scpdsi_lmr_subsample = handle_subsample['pdsi'].values
handle_subsample.close()
print('Subsample shape:'); print(scpdsi_lmr_subsample.shape)

years_lmr = time_lmr/365
years_lmr = years_lmr.astype(int)


### CALCULATIONS

# Reshape the array so that iterations and ensemble members are on the same axis.
ntime = scpdsi_lmr_subsample.shape[0]
niter = scpdsi_lmr_subsample.shape[1]
nlat  = scpdsi_lmr_subsample.shape[2]
nlon  = scpdsi_lmr_subsample.shape[3]
nens  = scpdsi_lmr_subsample.shape[4]
scpdsi_lmr_subsample_rolled = np.rollaxis(scpdsi_lmr_subsample,4,2)
scpdsi_lmr_subsample_reshape = np.reshape(scpdsi_lmr_subsample_rolled,(ntime,niter*nens,nlat,nlon))

# Compute the mean of the LMR iterations
scpdsi_lmr_mean = np.mean(scpdsi_lmr_allmeans,axis=1)

if atlas_name == 'DaiPDSI':
    #
    # Compute annual-mean values
    nyears_atlas = len(years_atlas)
    nlat_atlas   = len(lat_atlas)
    nlon_atlas   = len(lon_atlas)
    pdsi_atlas_monthly_2d = np.reshape(pdsi_atlas_monthly,(nyears_atlas,12,nlat_atlas,nlon_atlas))
    pdsi_atlas = np.zeros((nyears_atlas,nlat_atlas,nlon_atlas)); pdsi_atlas[:] = np.nan
    for j in range(nlat_atlas):
        print('Calculating annual mean DaiPDSI values: '+str(j+1)+'/'+str(nlat_atlas))
        for i in range(nlon_atlas):
            pdsi_atlas[:,j,i] = mpe.annual_mean(years_atlas,pdsi_atlas_monthly_2d[:,:,j,i])
    #
    # Regrid the DaiPDSI data from -180 to 180 longitude to 0 to 360 longitude
    indices_east = np.where(lon_atlas > 0)[0]
    indices_west = np.where(lon_atlas < 0)[0]
    #
    lon_atlas_east = lon_atlas[indices_east]
    lon_atlas_west = lon_atlas[indices_west]+360
    lon_atlas = np.concatenate((lon_atlas_east,lon_atlas_west),axis=0)
    #
    pdsi_atlas_east = pdsi_atlas[:,:,indices_east]
    pdsi_atlas_west = pdsi_atlas[:,:,indices_west]
    pdsi_atlas = np.concatenate((pdsi_atlas_east,pdsi_atlas_west),axis=2)
    #
elif (atlas_name == 'NADA') or (atlas_name == 'OWDA') or (atlas_name == 'MADA'):
    #
    # Reorder the pdsi_atlas variable to be time,lat,lon like the LMR reconstruction
    pdsi_atlas = np.swapaxes(pdsi_atlas,0,2)
    #
    # Replace all atlas values of -99.999 with nan.
    pdsi_atlas[pdsi_atlas<-90] = np.nan

# Define 2d versions of lat and lon
lon_atlas_2d,lat_atlas_2d = np.meshgrid(lon_atlas,lat_atlas)
lon_lmr_2d,  lat_lmr_2d   = np.meshgrid(lon_lmr,  lat_lmr)

# Regrid the atlas data set to match the LMR grid
# Declare a new variable for the regridded pdsi, with all grid cells set to nan initially.
pdsi_atlas_regrid = np.zeros((len(years_atlas),nlat,nlon)); pdsi_atlas_regrid[:] = np.nan

# Regrid the data to the grid of the model
for year in range(len(years_atlas)):
    pdsi_atlas_regrid[year,:,:] = basemap.interp(pdsi_atlas[year,:,:], lon_atlas, lat_atlas, lon_lmr_2d, lat_lmr_2d, order=1)

# For the DaiPDSI dataset, for lats poleward of 78N, set to nan, since the DaiPDSI dataset doesn't have data that far north.
if atlas_name == 'DaiPDSI':
    index_78N = np.where(lat_lmr==78)[0][0]
    pdsi_atlas_regrid[:,index_78N:,:] = np.nan



### COMPUTE CORRELATIONS
# Compute correlation between atlas PDSI and LMR scPDSI at every point

# Set up variables
scpdsi_correlation = np.zeros((nlat,nlon)); scpdsi_correlation[:] = np.nan
scpdsi_pvalue      = np.zeros((nlat,nlon)); scpdsi_pvalue[:]      = np.nan
scpdsi_R_squared   = np.zeros((nlat,nlon)); scpdsi_R_squared[:]   = np.nan
scpdsi_CE          = np.zeros((nlat,nlon)); scpdsi_CE[:]          = np.nan

# Select years 0-2000 from the drought atlas data.
if atlas_name == 'MADA':
    pdsi_atlas_regrid_2k = np.zeros((2001,nlat,nlon))
    pdsi_atlas_regrid_2k[:] = np.nan
    pdsi_atlas_regrid_2k[1300:2001,:,:] = pdsi_atlas_regrid[0:701,:,:]
elif atlas_name == 'DaiPDSI':
    pdsi_atlas_regrid_2k = np.zeros((2001,nlat,nlon))
    pdsi_atlas_regrid_2k[:] = np.nan
    pdsi_atlas_regrid_2k[1850:2001,:,:] = pdsi_atlas_regrid[0:151,:,:]
else:
    pdsi_atlas_regrid_2k = pdsi_atlas_regrid[0:2001,:,:]

# This is an meaningless variable used to find indexes which are not NaN in either dataset.
common_gridpoints = scpdsi_lmr_mean+pdsi_atlas_regrid_2k
common_gridpoints[~np.isnan(common_gridpoints)] = 1

pdsi_atlas_regrid_selected    = pdsi_atlas_regrid_2k*common_gridpoints
scpdsi_lmr_mean_selected      = scpdsi_lmr_mean*common_gridpoints
scpdsi_lmr_subsample_selected = scpdsi_lmr_subsample_reshape*common_gridpoints[:,None,:,:]

# Remove the mean of years 1951-1980
index_1951 = np.where(years_lmr==1951)[0][0]
index_1980 = np.where(years_lmr==1980)[0][0]
pdsi_atlas_regrid_selected    = pdsi_atlas_regrid_selected    - np.nanmean(pdsi_atlas_regrid_selected[index_1951:index_1980+1,:,:],axis=0)
scpdsi_lmr_mean_selected      = scpdsi_lmr_mean_selected      - np.nanmean(scpdsi_lmr_mean_selected[index_1951:index_1980+1,:,:],axis=0)
scpdsi_lmr_subsample_selected = scpdsi_lmr_subsample_selected - np.nanmean(np.nanmean(scpdsi_lmr_subsample_selected[index_1951:index_1980+1,:,:],axis=0),axis=0)

# Select the years of interest
index_min = np.where(years_lmr==years_min)[0][0]
index_max = np.where(years_lmr==years_max)[0][0]
pdsi_atlas_regrid_selected_time    = pdsi_atlas_regrid_selected[index_min:index_max+1,:,:]
scpdsi_lmr_mean_selected_time      = scpdsi_lmr_mean_selected[index_min:index_max+1,:,:]
scpdsi_lmr_subsample_selected_time = scpdsi_lmr_subsample_selected[index_min:index_max+1,:,:,:]
years_selected = years_lmr[index_min:index_max+1]

#i=130; j=65
for j in range(nlat):
    print('Computing statistics: '+str(j+1)+'/'+str(nlat))
    for i in range(nlon):
        #
        scpdsi_lmr_gridpoint = scpdsi_lmr_mean_selected_time[~np.isnan(scpdsi_lmr_mean_selected_time[:,j,i]),j,i]
        pdsi_atlas_gridpoint = pdsi_atlas_regrid_selected_time[~np.isnan(pdsi_atlas_regrid_selected_time[:,j,i]),j,i]
        #
        # Correlations
        #scpdsi_correlation[j,i] = np.corrcoef(scpdsi_lmr_gridpoint,pdsi_atlas_gridpoint)[0,1]
        scpdsi_correlation[j,i],scpdsi_pvalue[j,i] = pearsonr(scpdsi_lmr_gridpoint,pdsi_atlas_gridpoint)
        #
        # R-Squared
        scpdsi_R_squared[j,i] = (scpdsi_correlation[j,i])**2
        #
        # Coeffient of efficiency (CE)
        scpdsi_CE[j,i] = 1 - ( np.sum(np.power(pdsi_atlas_gridpoint.astype(float)-scpdsi_lmr_gridpoint,2),axis=0) / np.sum(np.power(pdsi_atlas_gridpoint.astype(float)-np.mean(pdsi_atlas_gridpoint.astype(float),axis=0),2),axis=0) )

# Compute average of PDSI from atlas and LMR for the entire region as well as the four regions used in Cook et al. 2014.
pdsi_atlas_mean_regions  = compute_regional_means.compute_US_means(pdsi_atlas_regrid_selected_time,lat_lmr,lon_lmr)
scpdsi_lmr_mean_regions  = compute_regional_means.compute_US_means(scpdsi_lmr_mean_selected_time,  lat_lmr,lon_lmr)
correlation_mean_regions = compute_regional_means.compute_US_means(np.expand_dims(scpdsi_correlation,0),lat_lmr,lon_lmr)

# Compute regions for every iteration of the LMR
nrealizations = scpdsi_lmr_subsample_selected.shape[1]
scpdsi_lmr_subsample_regions = {}
for i in range(nrealizations):
    print(' === Computing regional means: realization '+str(i+1)+'/'+str(nrealizations)+' ===')
    scpdsi_lmr_subsample_regions[i] = compute_regional_means.compute_US_means(scpdsi_lmr_subsample_selected_time[:,i,:,:],lat_lmr,lon_lmr)

# Reorganize the regional means.
subsample_lmr_mean_regions = {}
regions = scpdsi_lmr_subsample_regions[0].keys()
for region in regions:
    subsample_lmr_mean_regions[region] = np.zeros((nrealizations,len(years_selected))); subsample_lmr_mean_regions[region][:] = np.nan
    for i in range(nrealizations):
        subsample_lmr_mean_regions[region][i,:] = scpdsi_lmr_subsample_regions[i][region]

# Compute the verticies of the LMR grid, for plotting purposes
lon_lmr_wrap = copy.deepcopy(lon_lmr)
lon_lmr_wrap = np.insert(lon_lmr_wrap,0,lon_lmr_wrap[-1]-360)  # Add the right-most lon point to the left
lon_lmr_wrap = np.append(lon_lmr_wrap,lon_lmr_wrap[1]+360)     # Add the left-most lon point to the right
lon_lmr_edges = (lon_lmr_wrap[:-1] + lon_lmr_wrap[1:])/2

lat_lmr_edges = copy.deepcopy(lat_lmr)
lat_lmr_edges = (lat_lmr[:-1] + lat_lmr[1:])/2
lat_lmr_edges = np.insert(lat_lmr_edges,0,-90)  # Add the South Pole to the beginning
lat_lmr_edges = np.append(lat_lmr_edges,90)     # Add the North Pole to the end


### FIGURES
plt.style.use('ggplot')
# Mask out significant areas
pvalues_masked = np.ma.masked_less(scpdsi_pvalue,0.05)

if (atlas_name == 'NADA') or (atlas_name == 'DaiPDSI'):
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=65,llcrnrlon=220,urcrnrlon=305,resolution='c')
    figure_size = (11,11)
elif atlas_name == 'OWDA':
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=25,urcrnrlat=73,llcrnrlon=-14,urcrnrlon=47,resolution='c')
    figure_size = (8,11)
elif atlas_name == 'MADA':
    m = Basemap(projection='merc',lon_0=180,llcrnrlat=-10,urcrnrlat=58,llcrnrlon=58,urcrnrlon=147,resolution='c')
    figure_size = (11,11)

lon_lmr_edges_2d, lat_lmr_edges_2d = np.meshgrid(lon_lmr_edges, lat_lmr_edges)
x_lmr_edges, y_lmr_edges = m(lon_lmr_edges_2d,lat_lmr_edges_2d)
x_lmr, y_lmr = m(lon_lmr_2d,lat_lmr_2d)
levels = np.linspace(-1,1,21)

# FIGURE 1: Plot the correlation between atlas pdsi and LMR scpdsi
plt.figure(figsize=figure_size)
plt.axes([.05,.05,.9,.9])
m.pcolormesh(x_lmr_edges,y_lmr_edges,np.ma.masked_array(scpdsi_correlation,np.isnan(scpdsi_correlation)),cmap='RdBu_r',vmin=-1,vmax=1)
m.colorbar(location='bottom').ax.tick_params(labelsize=16)
m.pcolor(x_lmr_edges,y_lmr_edges,pvalues_masked,hatch='////',alpha=0)
m.drawcoastlines()
map_proxies.map_proxies(data_dir+experiment_name,m,'all','proxytypes',100,'b','k',1)
plt.title("(a) Correlation between "+atlas_name+" and LMR PDSI",fontsize=20)
if save_instead_of_plot:
    plt.savefig("figures/verification_lmr_vs_"+atlas_name+"_correlation_"+experiment_name[:20]+"_years_"+str(years_min)+"-"+str(years_max)+".png",dpi=300,format='png')
else:
    plt.show()


# FIGURE 2: Plot the R-squared values between atlas pdsi and LMR scpdsi
plt.figure(figsize=figure_size)
plt.axes([.05,.05,.9,.9])
m.pcolormesh(x_lmr_edges,y_lmr_edges,np.ma.masked_array(scpdsi_R_squared,np.isnan(scpdsi_R_squared)),cmap='RdBu_r',vmin=-1,vmax=1)
m.drawcoastlines()
m.colorbar(location='bottom').ax.tick_params(labelsize=16)
plt.title("R-squared between "+atlas_name+" and LMR PDSI",fontsize=20)
if save_instead_of_plot:
    plt.savefig("figures/verification_lmr_vs_"+atlas_name+"_R_squared_"+experiment_name[:20]+"_years_"+str(years_min)+"-"+str(years_max)+".png",dpi=300,format='png')
else:
    plt.show()


# FIGURE 3: Plot the coefficient of efficiency between atlas pdsi and LMR scpdsi
plt.figure(figsize=figure_size)
plt.axes([.05,.05,.9,.9])
m.pcolormesh(x_lmr_edges,y_lmr_edges,np.ma.masked_array(scpdsi_CE,np.isnan(scpdsi_CE)),cmap='RdBu_r',vmin=-1,vmax=1)
m.drawcoastlines()
m.colorbar(location='bottom').ax.tick_params(labelsize=16)
plt.title("(b) Coefficient of efficiency between "+atlas_name+" and LMR PDSI",fontsize=20)
if save_instead_of_plot:
    plt.savefig("figures/verification_lmr_vs_"+atlas_name+"_CE_"+experiment_name[:20]+"_years_"+str(years_min)+"-"+str(years_max)+".png",dpi=300,format='png')
else:
    plt.show()

"""
# This is a work in progress.

# Find the maximum values in the southwest U.S.
# Find the southwest U.S. region
j_indices = np.where((lat >= 0) & (lat <= 70))[0]          # All latitudes between 0 and 70N
i_indices = np.where((lon >= (360-80)) & (lon <= 360))[0]  # All longitudes between 80W and 0W
j_min = min(j_indices); j_max = max(j_indices)
i_min = min(i_indices); i_max = max(i_indices)
sst_mean_NAtl = mpe.spatial_mean(sst,lat,j_min,j_max,i_min,i_max)

print('Maximum values in the southwest U.S.')
scpdsi_correlation
"""

# FIGURE 4: Plot a time-series of atlas pdsi and LMR scpdsi for a particular region
if (atlas_name == 'NADA') or (atlas_name == 'DaiPDSI'):
    regions = ['entire','US','southwest','central','northwest','southeast']
elif atlas_name == 'MADA':
    regions = ['entire','MADA_good','MADA_bad']
else:
    regions = ['entire']

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)

for region in regions:
    #
    plt.figure(figsize=(15,4))
    window_size = 10
    plt.axes([.05,.15,.9,.75])
    pdsi_atlas_ts = pdsi_atlas_mean_regions[region][~np.isnan(pdsi_atlas_mean_regions[region])]
    pdsi_lmr_ts   = scpdsi_lmr_mean_regions[region][~np.isnan(scpdsi_lmr_mean_regions[region])]
    correlation_region = np.corrcoef(pdsi_atlas_ts,pdsi_lmr_ts)[0,1]
    #
    p025 = np.percentile(subsample_lmr_mean_regions[region],2.5,axis=0)
    p975 = np.percentile(subsample_lmr_mean_regions[region],97.5,axis=0)
    #
    if atlas_name == 'DaiPDSI':
        line1, = plt.plot(years_selected,pdsi_atlas_mean_regions[region],color='k',linewidth=2)
        line2 = plt.fill_between(years_selected.astype(int),p025,p975,color='b',alpha=0.2)
        line3, = plt.plot(years_selected,scpdsi_lmr_mean_regions[region],color='b',linewidth=2)
        plt.title("PDSI for the "+str(region)+" region, annual-mean. $R_{annual}$="+str('%1.2f' % correlation_region),fontsize=20)
    else:
        line1, = plt.plot(years_selected,np.convolve(pdsi_atlas_mean_regions[region],np.ones(window_size)/window_size,'same'),color='k',linewidth=2)
        line2 = plt.fill_between(years_selected.astype(int),np.convolve(p025,np.ones(window_size)/window_size,'same'),np.convolve(p975,np.ones(window_size)/window_size,'same'),color='b',alpha=0.2)
        line3, = plt.plot(years_selected,np.convolve(scpdsi_lmr_mean_regions[region],np.ones(window_size)/window_size,'same'),color='b',linewidth=2)
        plt.title("PDSI for the "+str(region)+" region, "+str(window_size)+"-year running mean. $R_{annual}$="+str('%1.2f' % correlation_region),fontsize=20)
    plt.xlim(years_min,years_max)
    plt.ylim(-4.5,4.5)
    legend = plt.legend([line1,line3],[atlas_name,"LMR"],loc=1,ncol=3,bbox_to_anchor=(1,-.06),prop={'size':14})
    legend.get_frame().set_alpha(0.5)
    plt.xlabel("Year",fontsize=16)
    plt.ylabel("PDSI",fontsize=16)
    if save_instead_of_plot:
        plt.savefig("figures/verification_lmr_vs_"+atlas_name+"_ts_"+experiment_name[:20]+"_"+str(region)+"_region_years_"+str(years_min)+"-"+str(years_max)+".png",dpi=300,format='png')
    else:
        plt.show()



### FIGURE FOR PAPER
# Plot maps and time series

plt.figure(figsize=(22,22))

# Panel 1: Map of correlations
ax1 = plt.subplot2grid((4,2),(0,0),rowspan=2)
m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=65,llcrnrlon=220,urcrnrlon=305,resolution='c',ax=ax1)
image1 = m.pcolormesh(x_lmr_edges,y_lmr_edges,np.ma.masked_array(scpdsi_correlation,np.isnan(scpdsi_correlation)),cmap='RdBu_r',vmin=-1,vmax=1)
m.colorbar(image1,location='bottom',pad=.25).ax.tick_params(labelsize=20)
m.pcolor(x_lmr_edges,y_lmr_edges,pvalues_masked,hatch='////',alpha=0)
m.drawparallels(np.arange(-90,91,20),labels=[True,False,False,False])
m.drawmeridians(np.arange(0,361,20), labels=[False,False,False,True])
m.drawcoastlines()
map_proxies.map_proxies(data_dir+experiment_name,m,'all','proxytypes',100,'b','k',1)
ax1.set_title("(a) R between "+atlas_name+" and LMR PDSI",fontsize=28,loc='left')

# Panel 2: Map of CE
ax2 = plt.subplot2grid((4,2),(0,1),rowspan=2)
m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=65,llcrnrlon=220,urcrnrlon=305,resolution='c',ax=ax2)
image2 = m.pcolormesh(x_lmr_edges,y_lmr_edges,np.ma.masked_array(scpdsi_CE,np.isnan(scpdsi_CE)),cmap='RdBu_r',vmin=-1,vmax=1)
m.colorbar(image2,location='bottom',pad=.25).ax.tick_params(labelsize=20)
m.drawparallels(np.arange(-90,91,20),labels=[True,False,False,False])
m.drawmeridians(np.arange(0,361,20), labels=[False,False,False,True])
m.drawcoastlines()
# Draw a box for the region of interest
lat_min = 32
lat_max = 40
lon_min = -125+360
lon_max = -105+360
x_region,y_region = m([lon_min,lon_min,lon_max,lon_max],[lat_min,lat_max,lat_max,lat_min])
xy_region = np.column_stack((x_region,y_region))
region_box = Polygon(xy_region,edgecolor='black',facecolor='none',linewidth=2,alpha=.5)
plt.gca().add_patch(region_box)
ax2.set_title("(b) CE between "+atlas_name+" and LMR PDSI",fontsize=28,loc='left')

# Panel 3 & 4: Time-series
letters = ['c','d']
if (atlas_name == 'NADA') or (atlas_name == 'DaiPDSI'): regions = ['US','southwest']
else:                                                   regions = ['entire','entire']
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
window_size = 10

for i,region in enumerate(regions):
    #
    if   i == 0: ax3 = plt.subplot2grid((4,2),(2,0),colspan=2)
    elif i == 1: ax3 = plt.subplot2grid((4,2),(3,0),colspan=2)
    pdsi_atlas_ts = pdsi_atlas_mean_regions[region][~np.isnan(pdsi_atlas_mean_regions[region])]
    pdsi_lmr_ts   = scpdsi_lmr_mean_regions[region][~np.isnan(scpdsi_lmr_mean_regions[region])]
    correlation_region = np.corrcoef(pdsi_atlas_ts,pdsi_lmr_ts)[0,1]
    #
    p025 = np.percentile(subsample_lmr_mean_regions[region],2.5,axis=0)
    p975 = np.percentile(subsample_lmr_mean_regions[region],97.5,axis=0)
    #
    if atlas_name == 'DaiPDSI':
        line1, = ax3.plot(years_selected,pdsi_atlas_mean_regions[region],color='k',linewidth=2)
        line2  = ax3.fill_between(years_selected.astype(int),p025,p975,color='b',alpha=0.2)
        line3, = ax3.plot(years_selected,scpdsi_lmr_mean_regions[region],color='b',linewidth=2)
        ax3.set_title("("+letters[i]+") PDSI for the "+str(region)+" region, annual-mean",fontsize=28,loc='left')
    else:
        line1, = ax3.plot(years_selected,np.convolve(pdsi_atlas_mean_regions[region],np.ones(window_size)/window_size,'same'),color='k',linewidth=2)
        line2  = ax3.fill_between(years_selected.astype(int),np.convolve(p025,np.ones(window_size)/window_size,'same'),np.convolve(p975,np.ones(window_size)/window_size,'same'),color='b',alpha=0.2)
        line3, = ax3.plot(years_selected,np.convolve(scpdsi_lmr_mean_regions[region],np.ones(window_size)/window_size,'same'),color='b',linewidth=2)
        ax3.set_title("("+letters[i]+") PDSI for the "+str(region)+" region, "+str(window_size)+"-year running mean",fontsize=28,loc='left')
    ax3.text(0.89,0.15,"$R_{annual}$="+str('%1.2f' % correlation_region),fontsize=20,transform=ax3.transAxes)
    ax3.set_xlim(years_min,years_max)
    ax3.set_ylim(-5,5)
    legend = ax3.legend([line1,line3],[atlas_name,"LMR"],loc=1,ncol=3,bbox_to_anchor=(1,.15),prop={'size':20})
    legend.get_frame().set_alpha(0.5)
    if i == 1: ax3.set_xlabel("Year",fontsize=20)
    ax3.set_ylabel("PDSI",fontsize=20)

if save_instead_of_plot:
    plt.savefig("figures/Fig2_lmr_vs_"+atlas_name+"_"+experiment_name[:20]+"_years_"+str(years_min)+"-"+str(years_max)+".png",dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()

regions = correlation_mean_regions.keys()
print(' ==== Mean of grid-point corrrelations in different regions ===') 
for region in regions:
    print(region+': '+str('%1.2f' % correlation_mean_regions[region]))
