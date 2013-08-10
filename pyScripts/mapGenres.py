from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
# don't plot features that are smaller than 1000 square km.
#Brazil - 14.23500, -51.92528
#-12.811801,-51.15921
map = Basemap(projection='poly', width=5000000., height=6000000. ,lat_0 = -14.2350, lon_0 = -51.92528,
              resolution = 'l', area_thresh = 100.)
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = '#eeefff')
#map.bluemarble()
# draw the edge of the map projection region (the projection limb)
map.drawmapboundary()
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))
plt.show()


