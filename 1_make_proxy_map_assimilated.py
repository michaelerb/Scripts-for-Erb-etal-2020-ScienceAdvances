#=======================================================================
# Map of the locations of assimilated proxy locations for an experiment.
#    author: Michael Erb
#    date  : May 24, 2017
#=======================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import glob

# Modify numpy load to allow pickles
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
#np.load = np_load_old

size       = 100
color      = 'k'
edgecolor  = 'k'
alpha      = 1
plotyear   = 'all'
#region     = 'all'
#proxy_type = 'all'

save_instead_of_plot = True
data_dir = '/projects/pd_lab/data/LMR/archive_output/'
experiment_name = 'productionFinal_gisgpcc_ccms4_LMRdbv0.4.0'
#experiment_name = 'LMR_gisgpcc_ccms4_LMRdbv0.4.0_NorthAmerica_only'
#experiment_name = 'LMR_gisgpcc_ccms4_LMRdbv0.4.0_NorthAmerica_exclude'
#experiment_name = 'LMR_gmst_FDR_screened_long_calib_Feb2018'


# LOAD DATA
handle = np.load(data_dir+experiment_name+'/r0/ensemble_mean_tas_sfc_Amon.npz')
lon = handle['lon']
lat = handle['lat']
handle.close()

# Loop through every directory, determining the lats and lons of proxies.
directories = glob.glob(data_dir+experiment_name+'/r*')

proxy_names = []
proxy_types = []
proxy_lats = []
proxy_lons = []

# Initialized variables to save the anmount of data points from year 0-2000
years = np.arange(0,2001)
nyears = years.shape[0]
years_bivalve      = np.zeros((nyears))
years_borehole     = np.zeros((nyears))
years_coral        = np.zeros((nyears))
years_document     = np.zeros((nyears))
years_ice          = np.zeros((nyears))
years_hybrid       = np.zeros((nyears))
years_lake         = np.zeros((nyears))
years_marine       = np.zeros((nyears))
#years_sclerosponge = np.zeros((nyears))
years_speleothem   = np.zeros((nyears))
years_tree         = np.zeros((nyears))
years_other        = np.zeros((nyears))

for iteration in range(len(directories)):
    # Load the assimilated proxies
    assimilated_proxies = np.load(directories[iteration]+'/assimilated_proxies.npy')
    #
    # Determine the names of all the assimilated proxies which exist for a given year.
    for i in range(len(assimilated_proxies)):
        proxy_type = list(assimilated_proxies[i].keys())[0]
        proxy_name = assimilated_proxies[i][proxy_type][0]
        #
        # Calculate how many proxies there are of each kind for each year.
        # If a proxy has already been loaded, don't load it again.
        if proxy_name not in proxy_names:
            #
            # Find the lat/lons for only the proxies for the given year.
            if ((plotyear == "all") or (plotyear in assimilated_proxies[i][proxy_type][3])):
                print('Getting metadata for proxy #'+str(len(proxy_names)+1)+': '+proxy_name)
                #print assimilated_proxies[i][proxy_type][0]
                proxy_names.append(proxy_name)
                proxy_types.append(proxy_type)
                proxy_lats.append(assimilated_proxies[i][proxy_type][1])
                proxy_lons.append(assimilated_proxies[i][proxy_type][2])
            #
            for index,year in enumerate(years):
                if year in assimilated_proxies[i][proxy_type][3]:
                    if   "bivalve"      in proxy_type.lower(): years_bivalve[index] += 1
                    elif "borehole"     in proxy_type.lower(): years_borehole[index] += 1
                    elif "coral"        in proxy_type.lower(): years_coral[index] += 1
                    elif "document"     in proxy_type.lower(): years_document[index] += 1
                    elif "ice"          in proxy_type.lower(): years_ice[index] += 1
                    elif "hybrid"       in proxy_type.lower(): years_hybrid[index] += 1
                    elif "lake"         in proxy_type.lower(): years_lake[index] += 1
                    elif "marine"       in proxy_type.lower(): years_marine[index] += 1
                    #elif "sclerosponge" in proxy_type.lower(): years_sclerosponge[index] += 1
                    elif "speleothem"   in proxy_type.lower(): years_speleothem[index] += 1
                    elif "tree"         in proxy_type.lower(): years_tree[index] += 1
                    else:                                      years_other[index] += 1



# FIGURES
plt.style.use('ggplot')
#plt.figure(figsize=(15,11))
plt.figure(figsize=(13,9.5))
plt.axes([.05,.05,.9,.9])

# Plot the proxy locations.
ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
m = Basemap(projection='robin',lon_0=0,resolution='c',ax=ax1)
x,y = m(proxy_lons,proxy_lats)

# Initialize variables for the proxy counts, lons, and lats.
#print np.unique(proxy_types)
n_bivalve,n_borehole,n_coral,n_document,n_ice,n_hybrid,n_lake,n_marine,n_sclerosponge,n_speleothem,n_tree,n_other=0,0,0,0,0,0,0,0,0,0,0,0
x_bivalve,x_borehole,x_coral,x_document,x_ice,x_hybrid,x_lake,x_marine,x_sclerosponge,x_speleothem,x_tree,x_other=[],[],[],[],[],[],[],[],[],[],[],[]
y_bivalve,y_borehole,y_coral,y_document,y_ice,y_hybrid,y_lake,y_marine,y_sclerosponge,y_speleothem,y_tree,y_other=[],[],[],[],[],[],[],[],[],[],[],[]

for i in range(len(proxy_lats)):
    if   "bivalve"      in proxy_types[i].lower(): n_bivalve      += 1; x_bivalve.append(x[i]);      y_bivalve.append(y[i])
    #elif "borehole"     in proxy_types[i].lower(): n_borehole     += 1; x_borehole.append(x[i]);     y_borehole.append(y[i])
    elif "coral"        in proxy_types[i].lower(): n_coral        += 1; x_coral.append(x[i]);        y_coral.append(y[i])
    #elif "document"     in proxy_types[i].lower(): n_document     += 1; x_document.append(x[i]);     y_document.append(y[i])
    elif "ice"          in proxy_types[i].lower(): n_ice          += 1; x_ice.append(x[i]);          y_ice.append(y[i])
    #elif "hybrid"       in proxy_types[i].lower(): n_hybrid       += 1; x_hybrid.append(x[i]);       y_hybrid.append(y[i])
    elif "lake"         in proxy_types[i].lower(): n_lake         += 1; x_lake.append(x[i]);         y_lake.append(y[i])
    #elif "marine"       in proxy_types[i].lower(): n_marine       += 1; x_marine.append(x[i]);       y_marine.append(y[i])
    #elif "sclerosponge" in proxy_types[i].lower(): n_sclerosponge += 1; x_sclerosponge.append(x[i]); y_sclerosponge.append(y[i])
    #elif "speleothem"   in proxy_types[i].lower(): n_speleothem   += 1; x_speleothem.append(x[i]);   y_speleothem.append(y[i])
    elif "tree"         in proxy_types[i].lower(): n_tree         += 1; x_tree.append(x[i]);         y_tree.append(y[i])
    else:                                          n_other        += 1; x_other.append(x[i]);        y_other.append(y[i])

m.drawcoastlines()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='whitesmoke',lake_color='white',zorder=0)
m.drawparallels(np.arange(-80,81,20))
m.drawmeridians(np.arange(-180,181,60))

# Make scatterplots of all points
m.scatter(x_bivalve     ,y_bivalve     ,size,marker=(6,1,0),facecolor='Gold'        ,edgecolor=edgecolor,alpha=alpha,label='Bivalves ('+str(n_bivalve)+')')
#m.scatter(x_borehole    ,y_borehole    ,size,marker=(6,1,0),facecolor='DarkKhaki'   ,edgecolor=edgecolor,alpha=alpha,label='Boreholes ('+str(n_borehole)+')')
m.scatter(x_coral       ,y_coral       ,size,marker='o'    ,facecolor='DarkOrange'  ,edgecolor=edgecolor,alpha=alpha,label='Corals and sclerosponges ('+str(n_coral)+')')
#m.scatter(x_document    ,y_document    ,size,marker='*'    ,facecolor='Black'       ,edgecolor=edgecolor,alpha=alpha,label='Documents ('+str(n_document)+')')
m.scatter(x_ice         ,y_ice         ,size,marker='d'    ,facecolor='LightSkyBlue',edgecolor=edgecolor,alpha=alpha,label='Glacier ice ('+str(n_ice)+')')
#m.scatter(x_hybrid      ,y_hybrid      ,size,marker=(8,2,0),facecolor='DeepSkyBlue' ,edgecolor=edgecolor,alpha=alpha,label='Hybrid ('+str(n_hybrid)+')')
m.scatter(x_lake        ,y_lake        ,size,marker='s'    ,facecolor='RoyalBlue'   ,edgecolor=edgecolor,alpha=alpha,label='Lake sediments ('+str(n_lake)+')')
#m.scatter(x_marine      ,y_marine      ,size,marker='s'    ,facecolor='SaddleBrown' ,edgecolor=edgecolor,alpha=alpha,label='Marine sediments ('+str(n_marine)+')')
#m.scatter(x_sclerosponge,y_sclerosponge,size,marker='o'    ,facecolor='Red'         ,edgecolor=edgecolor,alpha=alpha,label='Sclerosponges ('+str(n_sclerosponge)+')')
#m.scatter(x_speleothem  ,y_speleothem  ,size,marker='d'    ,facecolor='DeepPink'    ,edgecolor=edgecolor,alpha=alpha,label='Speleothems ('+str(n_speleothem)+')')
m.scatter(x_tree        ,y_tree        ,size,marker='^'    ,facecolor='LimeGreen'   ,edgecolor=edgecolor,alpha=alpha,label='Trees ('+str(n_tree)+')')
#m.scatter(x_other       ,y_other       ,size,marker='v'    ,facecolor='Black'       ,edgecolor=edgecolor,alpha=alpha,label='Other ('+str(n_other)+')')

box = ax1.get_position()
ax1.set_position([box.x0,box.y0,box.width*0.75,box.height])
proxylegend = plt.legend(loc='center left',scatterpoints=1,bbox_to_anchor=(1,.5))
proxylegend.get_frame().set_alpha(0)
ax1.set_title("(a) Proxy locations",fontsize=20)


ax2 = plt.subplot2grid((3,1),(2,0))
#years_data = np.vstack([years_bivalve,years_borehole,years_coral,years_document,years_ice,years_hybrid,years_lake,years_marine,years_speleothem,years_tree,years_other])
#labels = ['Bivalves','Boreholes','Corals and sclerosponges','Documents','Glacier ice','Hybrid','Lake sediments','Marine sediments','Speleothems','Trees','Other']
#colors = ['Gold','DarkKhaki','DarkOrange','Black','LightSkyBlue','DeepSkyBlue','RoyalBlue','SaddleBrown','DeepPink','LimeGreen','Black']

years_data = np.vstack([years_bivalve,years_coral,years_ice,years_lake,years_tree])
labels = ['Bivalves','Corals and sclerosponges','Glacier ice','Lake sediments','Trees']
colors = ['Gold','DarkOrange','LightSkyBlue','RoyalBlue','LimeGreen']

ax2.stackplot(years,years_data,colors=colors,labels=labels)
#ax2.legend(loc=2)
ax2.set_xlim(years[0],years[-1])
ax2.set_xlabel('Year C.E.')
ax2.set_ylabel('Number of records')
ax2.set_title("(b) Proxy coverage through time",fontsize=20)

if save_instead_of_plot:
    plt.savefig("figures/proxy_map_"+experiment_name+".png",dpi=300,format='png',bbox_inches='tight')
else:
    plt.show()


if n_other != 0: print('WARNING!!!  n_other is not equal to 0!  n_other = '+str(n_other))
