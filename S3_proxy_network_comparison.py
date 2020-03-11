#=============================================================================
# This script is simply to figure out how much of an overlap the LMR dataset
# has with the NADA dataset.
#    author: Michael P. Erb
#    date  : 7/25/2018
#=============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob

save_instead_of_plot = True

# Modify numpy load to allow pickles
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Open Ed Cook's list of NADA proxies
nada_records = np.genfromtxt('/home/mpe32/analysis/5_drought/nada_details/information/TRCRNS.LIST_1845',dtype='str',delimiter=[10,10,7,7,6,5,3,6,3,50],comments='ignorethis_jdlfkjsldkfj')
nada_state       = nada_records[:,0]
nada_proxyID     = nada_records[:,1]
nada_unknown     = nada_records[:,2]
nada_year_start  = nada_records[:,3]
nada_year_end    = nada_records[:,4]
nada_lat_whole   = nada_records[:,5].astype(np.int)
nada_lat_decimal = nada_records[:,6].astype(np.int)/60.
nada_lon_whole   = nada_records[:,7].astype(np.int)
nada_lon_decimal = nada_records[:,8].astype(np.int)/60.
nada_description = nada_records[:,9]

# Checking.  Some records have minute values above 60.  These locations should be double-checked and updated.
print(np.where(nada_records[:,6].astype(np.int) > 60))
print(np.where(nada_records[:,8].astype(np.int) > 60))
indices_check = np.where((nada_records[:,6].astype(np.int) > 60) | (nada_records[:,8].astype(np.int) > 60))[0]

# Remove the '\n' from the end of the descriptions
nproxies_nada = nada_records.shape[0]
for i in range(nproxies_nada):
    nada_description[i] = nada_description[i].replace('\n','')

# Combine whole and decimal values for lat and lon
nada_lat = nada_lat_whole + nada_lat_decimal
nada_lon = nada_lon_whole - nada_lon_decimal + 360
print('NADA lat range: '+str(min(nada_lat))+' - '+str(max(nada_lat)))
print('NADA lon range: '+str(min(nada_lon))+' - '+str(max(nada_lon)))

# Load the assimilated proxies for the LMR experiment
lmr_dir = '/projects/pd_lab/data/LMR/archive_output/'
lmr_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'

# Loop through every directory, determining the lats and lons of proxies.
directories = glob.glob(lmr_dir+lmr_name+'/r*')

proxy_names = []
proxy_types = []
proxy_lats = []
proxy_lons = []

for iteration in range(len(directories)):
    # Load the assimilated proxies
    assimilated_proxies = np.load(directories[iteration]+'/assimilated_proxies.npy')
    #
    # Determine the names of all the assimilated proxies which exist for a given year.
    for i in range(len(assimilated_proxies)):
        proxy_type = list(assimilated_proxies[i].keys())[0]
        proxy_name = assimilated_proxies[i][proxy_type][0]
        #
        # Save lats and lons.  If a proxy has already been loaded, don't load it again.
        if proxy_name not in proxy_names:
            print('Getting metadata for proxy #'+str(len(proxy_names)+1)+': '+proxy_name)
            proxy_names.append(proxy_name)
            proxy_types.append(proxy_type)
            proxy_lats.append(assimilated_proxies[i][proxy_type][1])
            proxy_lons.append(assimilated_proxies[i][proxy_type][2])

nproxies_lmr = len(proxy_names)

# Determine how many proxies are in the region shown
nproxies_in_region_lmr = 0
for i in range(nproxies_lmr):
    if (proxy_lats[i] >= 15) and (proxy_lats[i] <= 70) and (proxy_lons[i] >= 190) and (proxy_lons[i] <= 310):
        nproxies_in_region_lmr += 1


"""
# Look for a particular proxy
for i in range(nproxies_lmr):
    if (proxy_lats[i] > 51.5) and (proxy_lats[i] < 52.5) and (proxy_lons[i] > 280) and (proxy_lons[i] < 285):
        print(i)
"""

"""
# Look for proxies at the same lat/lon
for i in range(nproxies_lmr):
#for i in range(10):
    #
    # If a proxy is outside the region of the NADA proxies, ignore it. 
    if (proxy_lats[i] < 10) or (proxy_lats[i] > 75) or (proxy_lons[i] < 190) or (proxy_lons[i] > 310):
#        print(i,': outside region')
        pass
    #
    # Loook for proxies at a similar lat/lon
#    i = 4
    i = 875
#    i = 881
    print(' === LMR proxy '+str(i)+': '+proxy_names[i]+' | '+proxy_types[i]+' | '+str(proxy_lats[i])+' N | '+str(proxy_lons[i])+' E ===')
    for j in range(nproxies_nada):
        if (abs(proxy_lats[i]-nada_lat[j]) < .1) and (abs(proxy_lons[i]-nada_lon[j]) < .1):
            print('NADA proxy '+str(j)+': '+nada_proxyID[j]+' | '+nada_unknown[j]+' | '+str(nada_lat[j])+' N | '+str(nada_lon[j])+' E | '+nada_description[j])
"""



# FIGURES
plt.style.use('ggplot')
plt.figure(figsize=(15,11))
plt.axes([.05,.05,.9,.9])

# Plot the proxy locations.
ax1 = plt.subplot2grid((1,1),(0,0))
m = Basemap(projection='merc',lon_0=180,llcrnrlat=15,urcrnrlat=70,llcrnrlon=190,urcrnrlon=310,resolution='c')
x_nada,y_nada = m(nada_lon,  nada_lat)
x_lmr, y_lmr  = m(proxy_lons,proxy_lats)

m.drawcoastlines()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='whitesmoke',lake_color='white',zorder=0)
m.drawparallels(np.arange(-90,91,10),  labels=[True, False,False,False],fontsize=12)
m.drawmeridians(np.arange(-180,181,10),labels=[False,False,False,True ],fontsize=12)

scatter_nada = m.scatter(x_nada,y_nada,100,color='deepskyblue')
scatter_lmr  = m.scatter(x_lmr, y_lmr ,20, color='darkgreen')
#scatter_nada2 = m.scatter(x_nada[indices_check],y_nada[indices_check],300,color='b')

plt.legend([scatter_lmr,scatter_nada],['LMR proxies in region (n='+str(nproxies_in_region_lmr)+')','NADA proxies (n='+str(nproxies_nada)+')'],loc='lower left',scatterpoints=1,fontsize=16)
ax1.set_title("North American proxies used in LMR and NADA",fontsize=20)

if save_instead_of_plot:
    plt.savefig("figures/FigS3_proxy_nada_lmr_comparison.png",dpi=300,format='png',bbox_inches='tight')
else:
    plt.show()

