#Standard libraries
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#############Libraries for Map
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath



#Directory des fichiers
dire ="/home/delhaye/Documents/Travail Bac3/Temperature Europe/"
#Chemin du fichier
nc_file = dire+'tasmax_Amon_EC-Earth3_historical_r1i1p1f1_gr_merge_185001-201412.nc'
#Charge le fichier via Dataset
fh = Dataset(nc_file, mode='r')

print(fh)

tas_hist= fh.variables['tasmax'][:]
lon = fh.variables['lon'][:]
lat = fh.variables['lat'][:]
fh.close()

tas_hist.shape

#Charge le 2ème fichier (projection des températures 2015-2100)
nc_file = dire+'tasmax_Amon_EC-Earth3_ssp585_r1i1p1f1_gr_merge_201501-210012.nc'
fh = Dataset(nc_file, mode='r')
tas_ssp= fh.variables['tasmax'][:]
lon = fh.variables['lon'][:]
lat = fh.variables['lat'][:]
fh.close()

start_y=1850
end_y=2014

#Dimensions du vecteur tas_hist
nm,nx,ny=tas_hist.shape

#Temperature tous les mois d'aout sur la période
aout_hist = tas_hist[7::12,:,:]
aout_ssp = tas_ssp[7::12,:,:]

aout_hist.shape

#Moyenne de tous les mois d'août
aout_hist_m = np.mean(aout_hist,axis=0)
aout_ssp_m = np.mean(aout_ssp,axis=0)


### Map ###

#Echelle
scale=np.round(np.arange(-30,30.1,5),0)

#To avoid having blank  around 0°lon
lon[0]=0
lon[-1]=359.9
#Palette de couleurs 
palette=plt.cm.seismic


theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

lon_mesh,lat_mesh = np.meshgrid(lon,lat)

fig = plt.subplots(figsize=(6, 6), dpi=102)
plt.axis('off')
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0.0,central_latitude=90.0))
ylims = [30,90]
xlims = [-180,180]

ax.set_extent(xlims+ylims, crs=ccrs.PlateCarree())
ax.set_boundary(circle, transform=ax.transAxes)

ax.stock_img()


cs=ax.contourf(lon, lat, aout_hist_m-273.15,scale,transform=ccrs.PlateCarree(),cmap=palette,transform_first=False,extend="both")
ax.coastlines("110m", linewidth=0.5, color="black")
# Doing the gridlines we want
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,color="grey")
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,60))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,10))

#Colorbar en dessous de la carte
cbar = plt.colorbar(cs,orientation="horizontal")
#Légende
cbar.set_label('Max T in August $^\circ$C $_{Historical}$',fontsize=16)
#Taille de l'échelle
cbar.ax.tick_params(labelsize=12)
#Montre l'échelle que tous les 2 pas
cbar.set_ticks(scale[::2])
cbar.set_ticklabels(scale[::2])
#Titre
plt.title("Surface temperature")

#######Graph évolution T° en Belgique


##Fonction qui permet de trouver la valeur exacte d'une cellule d'un array la plus proche de "value"
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


find_nearest(lat, 50.8)

#pos_lat est la position de l'array à laquelle on veut la latitude
pos_lat = np.where(lat==(find_nearest(lat, 50.8)))[0][0]
pos_lat

#pos_lon comme pos_lat mais pour la longitude
pos_lon = np.where(lon==(find_nearest(lon, 4)))[0][0]
pos_lon

np.shape(aout_hist)

#Prendre les valeurs de T° moy pour la Belgique
tas_bel1=aout_hist[:,pos_lat,pos_lon]-273.15
tas_bel2=aout_ssp[:,pos_lat,pos_lon]-273.15
tas_bel2

#Mettre les 2 time series ensemble
tas_bel=np.append(tas_bel1,tas_bel2)
tas_bel

######Graph de l'évolution de la T°
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(np.arange(1850, 2015, 1),tas_bel1,'tab:blue',linestyle= '-')
ax.plot(np.arange(2015, 2101, 1),tas_bel2,'tab:red',linestyle= '-')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Max T in Belgium in EC-Earth3',fontsize=20)
plt.grid()
plt.xticks(np.arange(1850, 2101, 50),fontsize=18)
plt.yticks(np.arange(0, 35.1, 5),fontsize=18)


slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(2000, 2101, 1),tas_bel[-101:])
def predict(x):
   return slope * x + intercept
fitLine = predict(np.arange(2000, 2101, 1))
fitLine

######Graph de l'évolution de la T°
fig,ax = plt.subplots(figsize=(12,6))
ax.plot(np.arange(1850, 2015, 1),tas_bel1,'tab:blue',linestyle= '-')
ax.plot(np.arange(2015, 2101, 1),tas_bel2,'tab:red',linestyle= '-')
plt.plot(np.arange(2000, 2101, 1), fitLine, c='black')
plt.xlabel('Year',fontsize=20)
plt.ylabel('Max T in Belgium in EC-Earth3',fontsize=20)
plt.grid()
plt.xticks(np.arange(1850, 2101, 50),fontsize=18)
plt.yticks(np.arange(0, 35.1, 5),fontsize=18)


