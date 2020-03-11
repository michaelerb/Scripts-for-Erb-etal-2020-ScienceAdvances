#==============================================================================
# This script examines PDSI characteristics in the CESM LME set of simulations,
# many of which are single-forcing.  The goal is to examine if/how U.S. drought
# characteristics change under different forcings.  In particular, perform the
# following analyses:
#   Maps:
#     - Annual PDSI, means and variance
#     - Power spectra slopes
#   Southwest U.S.:
#     - Histogram
#     - Power spectra
#     - Drought lengths
#
#   author: Michael P. Erb
#   date  : 11/26/2019
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/general_lmr_analysis/python_functions')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import compute_regional_means
import seaborn as sns
import nitime.algorithms as tsa
import scipy.stats.distributions as dist
import statsmodels.api as sm
import scipy
from scipy import stats
import h5py

# Function to compute annual means
#var,days_per_month = pdsi_monthly,days_per_month
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
include_lmr  = True  # Include LMR in the analysis?
include_nada = True  # Include NADA in the analysis?
detrend_ts   = True  # Detrend the PDSI time series
data_dir = '/projects/pd_lab/data/models/CESM_LME/'
data_dir_pdsi = '/projects/pd_lab/data/models/CESM_LME/PDSI_from_NathanSteiger/'
experiments_firstiteration = ['GHG.001','LULC_HurttPongratz.001','ORBITAL.001','SSI_VSK_L.001','VOLC_GRA.001','002']
experiments = ['002',
               '003',
               '004',
               '005',
               '006',
               '007',
               '008',
               '009',
               '010',
               'GHG.001',
               'GHG.002',
               'GHG.003',
               'LULC_HurttPongratz.001',
               'LULC_HurttPongratz.002',
               'LULC_HurttPongratz.003',
               'ORBITAL.001',
               'ORBITAL.002',
               'ORBITAL.003',
               'SSI_VSK_L.001',
               'SSI_VSK_L.003',
               'SSI_VSK_L.004',
               'SSI_VSK_L.005',
               'VOLC_GRA.001',
               'VOLC_GRA.002',
               'VOLC_GRA.003',
               'VOLC_GRA.004',
               'VOLC_GRA.005']
region_bounds_sw = [32,40,-125,-105]  # 32-40N, 125-105W

# Load CESM LME data (PDSI from Nathan Steiger)
handle_basics = xr.open_dataset(data_dir+'b.e11.BLMTRC5CN.f19_g16.001.cam.h0.PDSI.085001-184912.nc',decode_times=False)
lat  = handle_basics['lat'].values
lon  = handle_basics['lon'].values
time = handle_basics['time'].values
handle_basics.close()
years = np.arange(850,1850)

# Compute annual means
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

# Load LMR data
data_dir_lmr    = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'

handle_allmeans = xr.open_dataset(data_dir_lmr+experiment_name+'/pdsi_MCruns_ensemble_mean.nc',decode_times=False)
scpdsi_lmr_allmeans = handle_allmeans['pdsi'].values
lon_lmr             = handle_allmeans['lon'].values
lat_lmr             = handle_allmeans['lat'].values
time_lmr            = handle_allmeans['time'].values
handle_allmeans.close()

years_lmr = time_lmr/365
years_lmr = years_lmr.astype(int)

# Load NADA data
handle_nada = xr.open_dataset('/projects/pd_lab/data/drought_atlases/NorthAmericanDroughtAtlas/LBDA2010/nada_hd2_cl.nc',decode_times=False)
pdsi_nada  = handle_nada['pdsi'].values
lat_nada   = handle_nada['lat'].values
lon_nada   = handle_nada['lon'].values
years_nada = handle_nada['time'].values
lon_nada   = lon_nada+360
handle_nada.close()


### FUNCTIONS

# Make a power spectra plot
def specplot(X,col,model,save_instead_of_plot):
    # standardize the index
    #scaler = StandardScaler()
    #scaler = scaler.fit(X)
    #Xn = scaler.transform(X).squeeze()
    sig = np.std(X)
    # carry out MTM
    Fs=1.0; adaptive=True; jackknife=False; NW=3
    f, psd_mt, nu = tsa.multi_taper_psd(X.squeeze(), Fs=Fs, adaptive=adaptive, jackknife=jackknife,NW=NW)
    #
    # chi^2 errors
    p975 = dist.chi2.ppf(.975, nu[0])
    p025 = dist.chi2.ppf(.025, nu[0])
    l1 = nu[0]/p975
    l2 = nu[0]/p025
    #
    #fit AR(1) model
    # ===============
    ar1_mod = sm.tsa.AR(X,missing='drop').fit(maxlag=1, trend='nc')
    gamma = ar1_mod.params[0]
    # Python requires us to specify the zero-lag value which is 1
    # Also note that the alphas for the AR model must be negated
    # We also set the betas for the MA equal to 0 for an AR(p) model
    ar = np.r_[1, -gamma]
    ma = np.r_[1, 0.0]
    sig_n = sig*np.sqrt(1-gamma**2) # noise standard deviation; designed to produce AR(1) with same variance as original
    #
    # compute noise spectra
    n_sim = 200; nf = len(f)
    n = len(X)
    ar1spec = np.empty([nf, n_sim])
    for k in range(0,n_sim):
        # simulate AR(1) noise
        ar1n = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=250,sigma=sig_n) 
        # compute its spectrum & store it
        fn, ar1spec[:,k], _ = tsa.multi_taper_psd(ar1n, Fs=Fs, adaptive=adaptive, jackknife=jackknife, NW=NW)
    #
    # estimate quantiles
    ar1_pct = np.percentile(ar1spec,[90,95,99],axis=1).T
    #
    # plot it all
    fig, ax = plt.subplots()
    #ax.plot(f,ar1_pct,color='k',linewidth=0.5,label=('90%','95%','99%'))
    ax.fill_between(f,psd_mt*nu[0]/p975, psd_mt*nu[0]/p025,alpha = 0.5,label=r'$\chi^2, \, 95\%$ HDR',color=col)
    ax.plot(f,psd_mt,color=col,label=model)
    ax.plot(f,ar1_pct[:,1],color='k',linewidth=0.5,label='AR(1), 95% threshold')
    legend = ax.legend(loc='lower left', shadow=False)
    # change axes to log log, and label periods
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tick_params(\
        axis='x',          # changes apply to the x-axis
        which='minor',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    #
    per = [500, 200, 100, 50, 20, 10, 5]
    xt = 1.0/np.array(per)
    ax.set_xticks(xt)
    ax.set_xticklabels(map(str, per))
    ax.set_ylim(ymax=200)
    ax.set_ylim(ymin=.05)
    # label axes
    ax.set_ylabel('Multitaper Spectral Density',fontsize=14)
    ax.set_xlabel('Period (years)',fontsize=14)
    ax.set_title('SW U.S. PDSI spectrum for '+model,fontsize=12)
    # save to file
    if save_instead_of_plot == True:
        plt.savefig('figures/pdsi_spectrum_'+model+'_ns_pdsi.png',dpi=300,format='png')
        plt.close()
    else:
        plt.show()

# Compute the length of droughts for every region.  Use the definition of droughts that Coats et al. 2013 did:
# Two years of PDSI < 0 indicate the start of a drought and two years of PDSI > 0 indicate the end.
#var = pdsi['002_annual_sw']
def drought_lengths(var):
    drought_length = []
    drought_start = np.nan
    #
    for i in range(var.shape[0]-1):
        if np.isnan(drought_start) and (var[i] < 0) and (var[i+1] < 0):
            drought_start = i
        if ~np.isnan(drought_start) and (var[i] > 0) and (var[i+1] > 0):
            drought_end = i
            drought_length.append(drought_end-drought_start)
            drought_start = np.nan
    #
    ge10 = sum(i >= 10 for i in drought_length)
    ge15 = sum(i >= 15 for i in drought_length)
    ge20 = sum(i >= 20 for i in drought_length)
    ge30 = sum(i >= 30 for i in drought_length)
    between5_9 = sum((i >= 5) and (i <= 9) for i in drought_length)
    between10_19 = sum((i >= 10) and (i <= 19) for i in drought_length)
    between20_29 = sum((i >= 20) and (i <= 29) for i in drought_length)
    return ge10,ge15,ge20,ge30,between5_9,between10_19,between20_29

# A function to determine whether a point is greater than 1 SD compared to a moving window
segment_length_oneside = 25
def below_1sd(var,index):
    #
    global segment_length_oneside
    nyears = var.shape[0]
    #
    seg_start = index-segment_length_oneside
    seg_end   = index+segment_length_oneside
    if seg_start < 0:        seg_start = 0;         seg_end = 50
    if seg_end   > nyears-1: seg_start = nyears-51; seg_end = nyears-1
    print(index,seg_start,seg_end)
    var_value   = var[index]               - np.mean(var[seg_start:seg_end+1])
    var_segment = var[seg_start:seg_end+1] - np.mean(var[seg_start:seg_end+1])
    #
    # Check to see if the value is above the threshold
    exceed_threshold_boolean = var_value < -1*np.std(var_segment)
    #
    return exceed_threshold_boolean

# Compute drought length vs average severity
#var = pdsi['002_annual_sw']; threshold = '1SD'
def length_and_severity(var,threshold):
    #
    # Save the indices of each drought
    drought_indices = {}
    drought_start   = np.nan
    drought_number  = 0
    nyears          = var.shape[0]
    if threshold == 0:
        for i in range(nyears-1):
            if np.isnan(drought_start) and (var[i] < 0) and (var[i+1] < 0):
                drought_start = i
                drought_indices[drought_number] = [i]
            elif ~np.isnan(drought_start) and (var[i] > 0) and (var[i+1] > 0):
                drought_start = np.nan
                drought_number += 1
            elif ~np.isnan(drought_start):
                drought_indices[drought_number].append(i)
    elif threshold == '1SD':
        for i in range(nyears-1):
            if np.isnan(drought_start) and below_1sd(var,i) and below_1sd(var,i+1):
                drought_start = i
                drought_indices[drought_number] = [i]
            elif ~np.isnan(drought_start) and ~below_1sd(var,i) and ~below_1sd(var,i+1):
                drought_start = np.nan
                drought_number += 1
            elif ~np.isnan(drought_start):
                drought_indices[drought_number].append(i)
    else:
        print('Specity a valid threshold in the length_and_severity function')
    #
    # Loop through each drought, calculating statistics
    ndroughts = len(drought_indices)
    lengths           = np.zeros(ndroughts); lengths[:]           = np.nan
    average_anomalies = np.zeros(ndroughts); average_anomalies[:] = np.nan
    total_anomalies   = np.zeros(ndroughts); total_anomalies[:]   = np.nan
    for i in range(ndroughts):
        indices = drought_indices[i]
        lengths[i]           = len(indices)
        average_anomalies[i] = np.mean(var[indices])
        total_anomalies[i]   = lengths[i]*average_anomalies[i]
    #
    return drought_indices,lengths,average_anomalies,total_anomalies



### CALCULATIONS

# Reshape the NADA dataset from lon-lat-years to years-lat-lon
pdsi_nada = np.swapaxes(pdsi_nada,0,2)

# Shorten LMR to cover the same years as the CESM simulations
years_lmr_indices = np.where((years_lmr >= years[0]) & (years_lmr <= years[-1]))[0]
scpdsi_lmr_allmeans_selected = scpdsi_lmr_allmeans[years_lmr_indices,:,:,:]

# Shorten NADA to cover the same years as the CESM simulations
years_nada_indices = np.where((years_nada >= years[0]) & (years_nada <= years[-1]))[0]
pdsi_nada_selected = pdsi_nada[years_nada_indices,:,:]

# Compute mean PDSI for the southwest U.S.
for experiment in experiments:
    pdsi[experiment+'_annual_sw'] = compute_regional_means.compute_means(pdsi[experiment+'_annual'],lat,lon,region_bounds_sw[0],region_bounds_sw[1],region_bounds_sw[2],region_bounds_sw[3])

# Compute mean PDSI for every iteration of the LMR
lmr_experiments = ['lmr01','lmr02','lmr03','lmr04','lmr05','lmr06','lmr07','lmr08','lmr09','lmr10','lmr11','lmr12','lmr13','lmr14','lmr15','lmr16','lmr17','lmr18','lmr19','lmr20']
for i in range(len(lmr_experiments)):
    pdsi[lmr_experiments[i]+'_annual_sw'] = compute_regional_means.compute_means(scpdsi_lmr_allmeans_selected[:,i,:,:],lat_lmr,lon_lmr,region_bounds_sw[0],region_bounds_sw[1],region_bounds_sw[2],region_bounds_sw[3])

# Compute mean PDSI for NADA
nada_experiments = ['nada']
pdsi[nada_experiments[0]+'_annual_sw'] = compute_regional_means.compute_means(pdsi_nada_selected,lat_nada,lon_nada,region_bounds_sw[0],region_bounds_sw[1],region_bounds_sw[2],region_bounds_sw[3])

# If LMR is to be included, add it to the list of experiments
if include_lmr:
    experiments = experiments + lmr_experiments
    experiments_firstiteration.append('lmr01')

if include_nada:
    experiments = experiments + nada_experiments
    experiments_firstiteration.append('nada')

# If specified, detrend the PDSI timeseries
if detrend_ts:
    for i in range(len(experiments)):
        pdsi_before_detrending = pdsi[experiments[i]+'_annual_sw']
        slope,intercept,r_value,p_value,std_err = stats.linregress(years,pdsi_before_detrending)
        pdsi_linear = slope*years + intercept
        pdsi_detrended = pdsi_before_detrending - pdsi_linear
        #plt.plot(years,pdsi_before_detrending,'k-',years,pdsi_linear,'b-')
        #
        pdsi[experiments[i]+'_annual_sw'] = pdsi_detrended

# Calculate the number of dry spells longer than a certain length
nexperiments = len(experiments)
ge10_regions         = np.zeros((nexperiments)); ge10_regions[:]         = np.nan
ge15_regions         = np.zeros((nexperiments)); ge15_regions[:]         = np.nan
ge20_regions         = np.zeros((nexperiments)); ge20_regions[:]         = np.nan
ge30_regions         = np.zeros((nexperiments)); ge30_regions[:]         = np.nan
between5_9_regions   = np.zeros((nexperiments)); between5_9_regions[:]   = np.nan
between10_19_regions = np.zeros((nexperiments)); between10_19_regions[:] = np.nan
between20_29_regions = np.zeros((nexperiments)); between20_29_regions[:] = np.nan

for i,experiment in enumerate(experiments):
    print(i,experiment)
    pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw']
    ge10_regions[i],ge15_regions[i],ge20_regions[i],ge30_regions[i],between5_9_regions[i],between10_19_regions[i],between20_29_regions[i] = drought_lengths(pdsi_total_sw_anomaly)

# Calculate the lengths and magnitudes of droughts
drought_indices   = {}
lengths           = {}
average_anomalies = {}
total_anomalies   = {}
for experiment in experiments:
    print(experiment)
    pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw']
    drought_indices[experiment],lengths[experiment],average_anomalies[experiment],total_anomalies[experiment] = length_and_severity(pdsi_total_sw_anomaly,'1SD')

# Compute correlations between all pairs of simulations for the first iterations
nexperiments_firstiteration = len(experiments_firstiteration)
correlations_firstiteration = np.zeros((nexperiments_firstiteration,nexperiments_firstiteration)); correlations_firstiteration[:] = np.nan
for i in range(nexperiments_firstiteration):
    for j in range(nexperiments_firstiteration):
        correlations_firstiteration[i,j] = np.corrcoef(pdsi[experiments_firstiteration[i]+'_annual_sw'],pdsi[experiments_firstiteration[j]+'_annual_sw'])[0,1]

# Compute correlations between all pairs of simulations
def correlations_between_pairs(pdsi,experiment_names1,experiment_names2):
    nexperiment_names1 = len(experiment_names1)
    nexperiment_names2 = len(experiment_names2)
    correlations = np.zeros((nexperiment_names1,nexperiment_names2)); correlations[:] = np.nan
    for i in range(nexperiment_names1):
        for j in range(nexperiment_names2):
            correlations[i,j] = np.corrcoef(pdsi[experiment_names1[i]+'_annual_sw'],pdsi[experiment_names2[j]+'_annual_sw'])[0,1]
    #
    return correlations

correlations = {}
experiment_names_all = ['002','003','004','005','006','007','008','009','010']
correlations['all']   = correlations_between_pairs(pdsi,experiment_names_all,experiment_names_all)
correlations['ghg']   = correlations_between_pairs(pdsi,experiment_names_all,['GHG.001','GHG.002','GHG.003'])
correlations['lulc']  = correlations_between_pairs(pdsi,experiment_names_all,['LULC_HurttPongratz.001','LULC_HurttPongratz.002','LULC_HurttPongratz.003'])
correlations['orbit'] = correlations_between_pairs(pdsi,experiment_names_all,['ORBITAL.001','ORBITAL.002','ORBITAL.003'])
correlations['ssi']   = correlations_between_pairs(pdsi,experiment_names_all,['SSI_VSK_L.001','SSI_VSK_L.003','SSI_VSK_L.004','SSI_VSK_L.005'])
correlations['volc']  = correlations_between_pairs(pdsi,experiment_names_all,['VOLC_GRA.001','VOLC_GRA.002','VOLC_GRA.003','VOLC_GRA.004','VOLC_GRA.005'])

# Remove duplicate and X:X correlations from the 'all' correlations
mask_all = np.ones_like(correlations['all'])
mask_all[np.triu_indices_from(mask_all)] = np.nan
correlations['all'] = correlations['all']*mask_all
correlations_all_flatten = correlations['all'].flatten()
correlations_all_flatten_nonan = correlations_all_flatten[np.isfinite(correlations_all_flatten)]

# Summarize the length and intensity stats from different experiments
def summarize_stats(lengths,experiments):
    global average_anomalies
    nexperiments = len(experiments)
    sims_number     = np.zeros((nexperiments)); sims_number[:]     = np.nan
    sims_avglength  = np.zeros((nexperiments)); sims_avglength[:]  = np.nan
    sims_avganomaly = np.zeros((nexperiments)); sims_avganomaly[:] = np.nan
    for i,experiment in enumerate(experiments):
        sims_number[i]     = len(lengths[experiment])
        sims_avglength[i]  = np.mean(lengths[experiment])
        sims_avganomaly[i] = np.mean(average_anomalies[experiment])
    #
    return sims_number, sims_avglength, sims_avganomaly

allsims_number,   allsims_avglength,   allsims_avganomaly   = summarize_stats(lengths,experiments)
firstsims_number, firstsims_avglength, firstsims_avganomaly = summarize_stats(lengths,experiments_firstiteration)

sims_number     = {}
sims_avglength  = {}
sims_avganomaly = {}
sims_number['full'],  sims_avglength['full'],  sims_avganomaly['full']  = summarize_stats(lengths,['002','003','004','005','006','007','008','009','010'])
sims_number['ghg'],   sims_avglength['ghg'],   sims_avganomaly['ghg']   = summarize_stats(lengths,['GHG.001','GHG.002','GHG.003'])
sims_number['lulc'],  sims_avglength['lulc'],  sims_avganomaly['lulc']  = summarize_stats(lengths,['LULC_HurttPongratz.001','LULC_HurttPongratz.002','LULC_HurttPongratz.003'])
sims_number['orbit'], sims_avglength['orbit'], sims_avganomaly['orbit'] = summarize_stats(lengths,['ORBITAL.001','ORBITAL.002','ORBITAL.003'])
sims_number['ssi'],   sims_avglength['ssi'],   sims_avganomaly['ssi']   = summarize_stats(lengths,['SSI_VSK_L.001','SSI_VSK_L.003','SSI_VSK_L.004','SSI_VSK_L.005'])
sims_number['volc'],  sims_avglength['volc'],  sims_avganomaly['volc']  = summarize_stats(lengths,['VOLC_GRA.001','VOLC_GRA.002','VOLC_GRA.003','VOLC_GRA.004','VOLC_GRA.005'])

# Calculate the means
meansims_number     = np.array([np.mean(sims_number['ghg']),    np.mean(sims_number['lulc']),    np.mean(sims_number['orbit']),    np.mean(sims_number['ssi']),    np.mean(sims_number['volc']),    np.mean(sims_number['full'])])
meansims_avglength  = np.array([np.mean(sims_avglength['ghg']), np.mean(sims_avglength['lulc']), np.mean(sims_avglength['orbit']), np.mean(sims_avglength['ssi']), np.mean(sims_avglength['volc']), np.mean(sims_avglength['full'])])
meansims_avganomaly = np.array([np.mean(sims_avganomaly['ghg']),np.mean(sims_avganomaly['lulc']),np.mean(sims_avganomaly['orbit']),np.mean(sims_avganomaly['ssi']),np.mean(sims_avganomaly['volc']),np.mean(sims_avganomaly['full'])])


# Calculate whether there are any significant differences between the means
if include_lmr:
    sims_number['lmr'],sims_avglength['lmr'],sims_avganomaly['lmr'] = summarize_stats(lengths,lmr_experiments)
    meansims_number     = np.append(meansims_number,    np.mean(sims_number['lmr']))
    meansims_avglength  = np.append(meansims_avglength, np.mean(sims_avglength['lmr']))
    meansims_avganomaly = np.append(meansims_avganomaly,np.mean(sims_avganomaly['lmr']))

# Calculate whether there are any significant differences between the means
if include_nada:
    sims_number['nada'],sims_avglength['nada'],sims_avganomaly['nada'] = summarize_stats(lengths,nada_experiments)
    meansims_number     = np.append(meansims_number,    np.mean(sims_number['nada']))
    meansims_avglength  = np.append(meansims_avglength, np.mean(sims_avglength['nada']))
    meansims_avganomaly = np.append(meansims_avganomaly,np.mean(sims_avganomaly['nada']))


# Organized correlation variables
names_correlations = ['All/All','All/GHG','All/LULC','All/Orbit','All/Solar','All/Volcanic']
correlations_all = [correlations_all_flatten_nonan,correlations['ghg'].flatten(),correlations['lulc'].flatten(),correlations['orbit'].flatten(),correlations['ssi'].flatten(),correlations['volc'].flatten()]

# Calculate 1-tailed t-test values
ncorrelations_all = len(correlations_all)
pvalues = np.zeros((ncorrelations_all)); pvalues[:] = np.nan
for i in range(ncorrelations_all):
    _,pvalues[i] = scipy.stats.ttest_1samp(correlations_all[i],0)



### FIGURES
plt.style.use('ggplot')
colors = list(plt.rcParams['axes.prop_cycle'])

# Plot boxplots of correlations between all ALL simulations and single-forcing simulations
f, ax = plt.subplots(1,1,figsize=(10,6),sharex=True)
boxplots = ax.boxplot(correlations_all,labels=names_correlations,patch_artist=True)
# Colors
colors_boxplots = ['skyblue','none','none','none','none','none','none']
for patch,color in zip(boxplots['boxes'],colors_boxplots):
    patch.set_facecolor(color)
ax.axhline(y=0,color='k',linewidth=1,linestyle='--',zorder=10)
ax.set_title('Correlations for southwest U.S. PDSI between individual simulations',fontsize=16)
ax.set_ylabel('Correlation',fontsize=16)
ax.set_xlabel('Experiments',fontsize=16)
ax.set_ylim(-.15,.15)
ax.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout()
for i in range(len(correlations_all)):
    #
    # If the mean is different from zero according to a 1-sample t-test, place an asterisk above the boxplot
    if pvalues[i] < 0.05:
        ax.text(i+1,.14,'*',horizontalalignment='center',verticalalignment='center',color='r',fontsize=16)
    #
    # Write the number of samples at th bottom of the plot
    ax.text(i+1,-.142,'n='+str(len(correlations_all[i])),horizontalalignment='center',verticalalignment='center',color='0.3',fontsize=12)

if save_instead_of_plot == True:
    plt.savefig('figures/FigS9_boxplot_correlations_ns_pdsi.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()


# Plot the correlations between all pairs of simulations
#names = ['All','GHG','LULC','Orbit','Solar','Volcanic','None']
names = ['GHG','LULC','Orbit','Solar','Volc.','All']
if include_lmr:  names = names + ['LMR']
if include_nada: names = names + ['NADA']

mask = np.zeros_like(correlations_firstiteration, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12,10))
sns.heatmap(correlations_firstiteration,mask=mask,center=0,square=True,cmap='coolwarm',vmin=-.1,vmax=.1,linewidths=1)
plt.xticks(np.arange(nexperiments_firstiteration)+.5,(names),fontsize=14)
plt.yticks(np.arange(nexperiments_firstiteration)+.5,(names),fontsize=14)
for j in range(nexperiments_firstiteration):
    for i in range(j,nexperiments_firstiteration):
        plt.text(i+.5,j+.5,str('%1.2f' % correlations_firstiteration[i,j]),horizontalalignment='center',verticalalignment='center')

plt.title('Correlations between southwest U.S. PDSI in first ensemble\nmember of CESM LME simulations and reconstructions',fontsize=18)
if save_instead_of_plot == True:
    plt.savefig('figures/correlations_pdsi_sw_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



# Plot time series of southwest U.S. annual-mean PDSI in each simulation
letters = ['a','b','c','d','e','f','g','h']
f, ax = plt.subplots(nexperiments_firstiteration,1,figsize=(12,12),sharex=True,sharey=True)
ax = ax.ravel()

for i,experiment in enumerate(experiments_firstiteration):
#    pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw'] - np.mean(pdsi[experiment+'_annual_sw'])
    pdsi_total_sw_anomaly = pdsi[experiment+'_annual_sw']
    #
    ax[i].plot(years,pdsi_total_sw_anomaly,color='k',linewidth=1)
    ax[i].plot(years,pdsi_total_sw_anomaly*0,'--',color='k')
    #
    # Indicate droughts
    for j in range(len(drought_indices[experiment])):
        indices = drought_indices[experiment][j]
        ax[i].axvspan(years[indices[0]]-.5,years[indices[-1]]+.5,color='sandybrown',alpha=.6,lw=0)
    #
    ax[i].set_title('('+letters[i]+') '+names[i],loc='left',fontsize=16)
    ax[i].set_xlim(years[0],years[-1])
    ax[i].set_ylim(-7.5,7.5)
    ax[i].set_ylabel('PDSI')
    if i == nexperiments_firstiteration-1:  ax[i].set_xlabel('Year')

plt.xticks(np.arange(900,1900,100))
if detrend_ts: plt.suptitle('Annual-mean southwest U.S. PDSI, detrended',fontsize=22)
else:          plt.suptitle('Annual-mean southwest U.S. PDSI',fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=.93)
if save_instead_of_plot == True:
    plt.savefig('figures/FigS10_pdsi_sw_ts_allsims_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()



# Make bar plots showing drought statistics
f, ax = plt.subplots(1,3,figsize=(15,3.5),sharex=True)
ax = ax.ravel()

nbars = len(meansims_number)
colors1 = [colors[0]['color']]*nbars
colors2 = [colors[0]['color']]*nbars
colors3 = [colors[0]['color']]*nbars
if include_lmr is not include_nada: colors1[-1] = colors[1]['color']; colors2[-1] = colors[1]['color']; colors3[-1] = colors[1]['color']
if include_lmr and include_nada:
    colors1[-2] = colors[1]['color']; colors2[-2] = colors[1]['color']; colors3[-2] = colors[1]['color']
    colors1[-1] = colors[3]['color']; colors2[-1] = colors[3]['color']; colors3[-1] = colors[3]['color']

ax[0].bar(np.arange(nexperiments_firstiteration),meansims_number,    tick_label=names,color=colors1)
ax[1].bar(np.arange(nexperiments_firstiteration),meansims_avglength, tick_label=names,color=colors2)
ax[2].bar(np.arange(nexperiments_firstiteration),meansims_avganomaly,tick_label=names,color=colors3)

keys_selected = ['ghg','lulc','orbit','ssi','volc','full']
if include_lmr: keys_selected  = keys_selected + ['lmr']
if include_nada: keys_selected = keys_selected + ['nada']
for i,key in enumerate(keys_selected):
    nvalues = len(sims_number[key])
    ax[0].plot(i*(np.zeros(nvalues)+1),sims_number[key],    'k_')
    ax[1].plot(i*(np.zeros(nvalues)+1),sims_avglength[key], 'k_')
    ax[2].plot(i*(np.zeros(nvalues)+1),sims_avganomaly[key],'k_')

ax[2].invert_yaxis()

ax[0].set_title('(a) Number of droughts',loc='left',fontsize=16)
ax[1].set_title('(b) Average drought length',loc='left',fontsize=16)
ax[2].set_title('(c) Average drought strength',loc='left',fontsize=16)

ax[0].set_ylabel('# of droughts')
ax[1].set_ylabel('Length (years)')
ax[2].set_ylabel('Strength (PDSI)')

if detrend_ts: plt.suptitle('Statistics of annual-mean southwest U.S. PDSI, detrended',fontsize=22)
else:          plt.suptitle('Statistics of annual-mean southwest U.S. PDSI',fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=.8)
if save_instead_of_plot == True:
    plt.savefig('figures/Fig6_pdsi_sw_stats_allsims_alliterations_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=300,format='png')
    plt.close()
else:
    plt.show()



# Figures below this line are not used in the paper, and have not been double-checked for accuracy.



# Plot the relationship between drought length and total severity.
f, ax = plt.subplots(2,5,figsize=(20,10),sharex=True,sharey=True)
ax = ax.ravel()
for i,experiment in enumerate(experiments_firstiteration):
    ax[i].scatter(lengths[experiment],total_anomalies[experiment],color='k')
    ax[i].set_title(experiment,fontsize=16)
    ax[i].set_xlabel('Lengths (years)')
    ax[i].set_ylabel('Total anomalies')

plt.suptitle('Drought lengths vs. total anomalies',fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=.9)
if save_instead_of_plot == True:
    plt.savefig('figures/pdsi_dryspell_scatter_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



min_total_anom = 0
for experiment in experiments_firstiteration:
    min_exp = np.min(total_anomalies[experiment])
    if min_exp < min_total_anom: min_total_anom = min_exp

min_total_anom = np.floor(min_total_anom)

# Plot the relationship between drought length and total severity.
f, ax = plt.subplots(2,5,figsize=(20,10),sharex=True,sharey=True)
ax = ax.ravel()
for i,experiment in enumerate(experiments_firstiteration):
    ax[i].hist(total_anomalies[experiment],bins=np.arange(min_total_anom,1,1))
    ax[i].set_title(experiment,fontsize=16)
    ax[i].set_xlabel('Total drought anomaly')
    ax[i].set_ylabel('Frequency')

plt.suptitle('Total drought anomalies (average anomaly times length)',fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=.9)
if save_instead_of_plot == True:
    plt.savefig('figures/pdsi_dryspell_hist_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



# Kernel density estimate of pdsi.
f, ax = plt.subplots(1,1,figsize=(12,8))

for i,experiment in enumerate(experiments_firstiteration):
    sns.kdeplot(total_anomalies[experiment],shade=False,ax=ax,label=names[i]+' (mean = '+str('%1.2f' % np.mean(total_anomalies[experiment]))+')')

ax.set_xlabel('Total drought anomaly')
ax.set_ylabel('Frequency')
ax.set_ylim(0,.15)
f.suptitle('Kernel density estimate for total drought anomalies for\nsouthwest U.S. droughts over 850-1849 in different CESM LME experiments',fontsize=20)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot:
    plt.savefig('figures/pdsi_dryspell_kde_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



# Kernel density estimate of pdsi.
f, ax = plt.subplots(1,1,figsize=(12,8))

for i,experiment in enumerate(experiments_firstiteration):
    sns.kdeplot(pdsi[experiment+'_annual_sw'],shade=False,ax=ax,label=names[i]+' (mean = '+str('%1.2f' % np.mean(pdsi[experiment+'_annual_sw']))+')')

ax.set_xlabel('Annual-mean southwest U.S. PDSI',fontsize=14)
ax.set_ylabel('Frequency')
if include_lmr: ax.set_ylim(0,.45)
else:           ax.set_ylim(0,.25)
f.suptitle('Kernel density estimate for southwest U.S. annual PDSI\nover 850-1849 in different CESM LME experiments, first iteration',fontsize=20)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot:
    plt.savefig('figures/pdsi_sw_kde_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=100,format='png')
    plt.close()
else:
    plt.show()



# Make power spectra plots
for experiment in experiments_firstiteration:
    specplot(pdsi[experiment+'_annual_sw'],'k',experiment,save_instead_of_plot)



# FIGURE: Compare the lengths of dry spells.
colors_plot = ['r','purple','g','c','m','y','orange']
if include_lmr is not include_nada: colors_plot = colors_plot + ['deepskyblue']
if include_lmr and include_nada: colors_plot    = colors_plot + ['deepskyblue'] + ['gold']

f, ax = plt.subplots(1,1,figsize=(8,8))
for i,experiment in enumerate(experiments_firstiteration):
    j = experiments.index(experiment)
    ax.plot(np.arange(3),[between5_9_regions[j],between10_19_regions[j],ge20_regions[j]],color=colors_plot[i],alpha=0.5,marker='o',linewidth=2,label=names[i])

ax.set_ylim(0,40)
ax.legend()
plt.xticks(np.arange(3),('5-9 years','10-19 years','>20 years'))
f.suptitle('Number of dry spells of different lengths\nin CESM LME experiments, first iteration',fontsize=16)
f.tight_layout()
f.subplots_adjust(top=.9)
if save_instead_of_plot:
    plt.savefig('figures/pdsi_dry_spell_lengths_lmr_'+str(include_lmr)+'_ns_pdsi.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()



"""
# Testing the width of axvspan
f, ax0 = plt.subplots(1,1,figsize=(12,4),sharex=True,sharey=True)
ax0.plot(np.arange(100),np.arange(100))
for i in range(100):
    if i % 2 == 0:
        ax0.axvspan(i-.5,i+.5,color='sandybrown',alpha=.6,lw=0)
        
plt.show()
"""
