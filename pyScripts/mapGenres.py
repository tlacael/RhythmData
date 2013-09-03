#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from mpl_toolkits.basemap import Basemap
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from math import sqrt
# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
# don't plot features that are smaller than 1000 square km.
#Brazil - 14.23500, -51.92528
#-12.811801,-51.15921
genres=['Axe', 'Bachata',
        'Bolero','Forro', 
        'Gaucha','Merengue',
        'Pagode','Salsa',
        'Sertaneja','Tango']

genresPrint=[u'Axé', 'Bachata',
        'Bolero',u'Forró', 
        u'Gaúcha','Merengue',
        'Pagode','Salsa',
        'Sertaneja','Tango']


colors = ['#00ffff', '#000000', '#ffff00',
            '#ff00ff', '#7fff00', '#ff8c00',
            '#eedd82', '#ffffff', '#ff0000', '#ffff00']

class mapTracks():
    
    def __init__(self, artistDict=[], songList=[]):
        '''
        #artistDic should be dict in format:
        # 'artistName':[genre, region, [lat,long], numSongs]
        # e.g.  'Zona Roja': [u'Merengue','Asuncion (Paraguay)', [-25.2821972, -57.63509999999999], 25]
       
        format for songList is ['title', 'actualGenre', 'predGenre', 'filepath', 'artist', 'region', [lat,long]]
        ['Araketo-Ao-Vivo-03-Pra-Levantar-Poeira-.mp3',
        'Axe',
        'Axe',
        '/Users/Tlacael/NYU/Latin_Music_Data/LMD/Axe/Axe\xcc\x81/Araketo - Ao Vivo - 03 - Pra Levantar Poeira.mp3',
        u'Araketo',
        u'Salvador - Bahia, Brazil',
        [-12.9703817, -38.512382]]
        '''
        artistDict = pickle.load(open(u'/Users/Tlacael/NYU/RhythmData/pyScripts/artistDict.pkl','r'))

        self.lat=[]
        self.lon=[]
        self.genre=[]
        self.predGenre=[]
        self.regions={}
        self.numSongs=[]
        self.curRegions=[]
        if artistDict:
            for key in artistDict:
                item = artistDict[key]
                if item[2]!='NA':
                    self.genre.append(genres.index(item[0]))
                    self.numSongs.append(item[3]) 
                    self.curRegions.append(item[1])
                    try:
                        prev=self.regions[item[1]][2]
                    except:
                        prev=0
                    self.regions[item[1]]=[item[2], prev+item[3]]
                    self.lat.append(item[2][0])
                    self.lon.append(item[2][1])

        if songList:
            for item in songList:
                if item[6]!='NA':
                    self.genre.append(genres.index(item[1]))
                    self.predGenre.append(genres.index(item[2]))
                    self.curRegions.append(item[5])
                    try:
                        prev = self.regions[item[5]][2]
                    except:
                        prev=0

                    self.regions[item[5]]=[item[6], prev+1]
                    self.lat.append(item[6][0])
                    self.lon.append(item[6][1])

        temp=[]
        for r in self.regions.items():
            temp.append([r[1][1], r])
        temp=sorted(temp, reverse=True)

        self.regions=[]
        for t in temp:
            self.regions.append(t[1:][0])

        self.artistDict=artistDict
        self.songList = songList

    def mapit(self,genreToMap=range(10), compareList=range(10)):
        '''
        m = Basemap(projection='poly', width=6000000., height=6000000. ,lat_0 = -14.2350, lon_0 = -51.92528,
                      resolution = 'l', area_thresh = 1000.)
        '''

        for i in genreToMap:
            try:
                compareList.pop(i)
            except:
                pass

        m = Basemap(width=12000000*2,height=9000000*2,projection='lcc',lat_0 = -14.2350, lon_0 = -51.92528,
                      resolution = 'l', area_thresh = 1000.)

        # draw coastlines, country boundaries, fill continents.
        lat = np.array(self.lat)
        lon = np.array(self.lon)
        jiggerx = np.random.rand(len(lat), 1)
        jiggerx = jiggerx.reshape(len(lat))
        jiggery = np.random.rand(len(lat), 1)
        jiggery = jiggery.reshape(len(lat))
        lat = lat + abs(jiggerx)
        lon = lon + abs(jiggery)
        x,y = m(lon,lat)
        fig1 = plt.figure()
        '''
        plot actual genres
        '''
        ax = fig1.add_subplot(111)
        m.ax = ax
        #z=uniform(len(x))
        #z = uniform(0,100,size=len(x[0:10]))
        #m.scatter(x[0:10],y[0:10],25,color='r',cmap=plt.cm.jet,marker='o',edgecolors='none',zorder=10)
        #m.colorbar(pad='8%')
        handles=[]
        labels=[]
        regionsToTitle=[]
        for i in genreToMap:
            tempx=np.array([x[idx] for idx in range(len(x)) if self.genre[idx]==i])
            tempy=np.array([y[idx] for idx in range(len(y)) if self.genre[idx]==i])
            tempReg=np.array([self.curRegions[idx] for idx in range(len(x)) if self.genre[idx]==i])

            if self.numSongs:
                tempSize = np.array([self.numSongs[idx] for idx in range(len(self.numSongs)) if self.genre[idx]==i], dtype=float)

                tempSize = ((tempSize/tempSize.max())+5/20.)*20
            else:
                tempSize = np.ones(len(x))*5
            print tempSize
            #m.plot([tempx[j] for j in range(len(tempx))],[tempy[j] for j in range(len(tempx))], 
            for idx in range(len(tempx)):
                regionsToTitle.append(tempReg[idx])
                h,=m.plot(tempx[idx], tempy[idx],
                        'o', 
                        color=colors[i], 
                        alpha=0.8,
                        markersize=tempSize[idx],
                        label=genresPrint[i])
            handles.append(h)
            labels.append(genresPrint[i])

        m.drawcoastlines()
        m.drawcountries(linewidth=0.3)
        m.fillcontinents(color = '#eeefff')
        regions = self.regions
        self.plottedList=[]
        thresh2=904300
        thresh1=304300
        latThresh = 80000
        for item in self.regions:
            if item[0] in regionsToTitle:
                xpt, ypt = m(item[1][0][1],item[1][0][0])
                closest=self.tooClose2([xpt,ypt])
                closestLat, drc =self.tooClose1(ypt)
                mve=90000
                mve2=250000
                if closest > thresh1:
                    if closestLat < latThresh:
                        print 'lat thresh', item[0]
                        print closestLat, 'drc', drc
                        #ax.text(xpt+1000,ypt+drc*mve2,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt), xytext=None, xycoords='data',textcoords='data') 
                        self.plottedList.append([xpt,ypt+drc*mve2])
                    else:
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt), xytext=None, xycoords='data',textcoords='data') 
                        #ax.text(xpt+1000,ypt+mve,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        self.plottedList.append([xpt,ypt+mve])
                '''
                elif closest > thresh2:
                    print item[0]
                    ax.text(xpt+1000,ypt-50000,item[0].encode('ASCII', 'ignore'), fontsize=8) 
                    self.plottedList.append([xpt,ypt])
                '''

        #map.bluemarble()
        # draw the edge of the map projection region (the projection limb)
        m.drawmapboundary()
        # draw lat/lon grid lines every 30 degrees.
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))
        #handles, labels = m.ax.get_legend_handles_labels()
        l=m.ax.legend(handles,labels);
        ls=l.get_lines()
        for lines in ls:
            lines._legmarker.set_ms(8) 
        rect = fig1.patch
        rect.set_facecolor('white')


       

        '''
        plot predicted genres
        '''
        '''
        ax = fig1.add_subplot(111)
        m.ax = ax
        #z=uniform(len(x))
        #z = uniform(0,100,size=len(x[0:10]))
        #m.scatter(x[0:10],y[0:10],25,color='r',cmap=plt.cm.jet,marker='o',edgecolors='none',zorder=10)
        #m.colorbar(pad='8%')
        handles=[]
        labels=[]
        regionsToTitle=[]
        for i in genreToMap:
            tempx=np.array([x[idx] for idx in range(len(x)) if self.genre[idx]==i])
            tempy=np.array([y[idx] for idx in range(len(y)) if self.genre[idx]==i])
            tempReg=np.array([self.curRegions[idx] for idx in range(len(x)) if self.genre[idx]==i])
            predGenre = [self.predGenre[idx] for idx in range(len(self.predGenre)) if self.genre[idx]==i]

            if self.numSongs:
                tempSize = np.array([self.numSongs[idx] for idx in range(len(self.numSongs)) if self.genre[idx]==i], dtype=float)

                tempSize = ((tempSize/tempSize.max())+5/20.)*20
            else:
                tempSize = np.ones(len(x))*5
            print tempSize
            #m.plot([tempx[j] for j in range(len(tempx))],[tempy[j] for j in range(len(tempx))], 
            for idx in range(len(tempx)):
                if predGenre[idx] in compareList:
                    regionsToTitle.append(tempReg[idx])
                    h,=m.plot(tempx[idx], tempy[idx],
                            'o', 
                            color=colors[i], 
                            alpha=0.8,
                            markersize=tempSize[idx],
                            label=genres[i])
                handles.append(h)
                labels.append(genres[i])

        m.drawcoastlines()
        m.drawcountries(linewidth=0.3)
        m.fillcontinents(color = '#eeefff')
        regions = self.regions
        self.plottedList=[]
        thresh2=904300
        thresh1=304300
        latThresh = 80000
        for item in self.regions:
            if item[0] in regionsToTitle:
                xpt, ypt = m(item[1][0][1],item[1][0][0])
                closest=self.tooClose2([xpt,ypt])
                closestLat, drc =self.tooClose1(ypt)
                mve=90000
                mve2=250000
                if closest > thresh1:
                    if closestLat < latThresh:
                        print 'lat thresh', item[0]
                        print closestLat, 'drc', drc
                        #ax.text(xpt+1000,ypt+drc*mve2,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt), xytext=None, xycoords='data',textcoords='data') 
                        self.plottedList.append([xpt,ypt+drc*mve2])
                    else:
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt), xytext=None, xycoords='data',textcoords='data') 
                        #ax.text(xpt+1000,ypt+mve,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        self.plottedList.append([xpt,ypt+mve])
                #'#''
                elif closest > thresh2:
                    print item[0]
                    ax.text(xpt+1000,ypt-50000,item[0].encode('ASCII', 'ignore'), fontsize=8) 
                    self.plottedList.append([xpt,ypt])
                #''#'

        #map.bluemarble()
        # draw the edge of the map projection region (the projection limb)
        m.drawmapboundary()
        # draw lat/lon grid lines every 30 degrees.
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))
        #handles, labels = m.ax.get_legend_handles_labels()
        m.ax.legend(handles,labels);
        rect = fig1.patch
        rect.set_facecolor('white')

        '#''
        for i in genreToMap:
            tempx=np.array([x[idx] for idx in range(len(x)) if self.genre[idx]==i])
            tempy=np.array([y[idx] for idx in range(len(y)) if self.genre[idx]==i])
            predGenre = [self.predGenre[idx] for idx in range(len(self.predGenre)) if self.genre[idx]==i]
            tempReg=np.array([self.curRegions[idx] for idx in range(len(x)) if self.genre[idx]==i])
            if self.numSongs:
                tempSize=np.array([self.numSongs[idx] for idx in range(len(self.numSongs)) if self.genre[idx]==i], dtype=float)
                tempSize = ((tempSize/tempSize.max())+5/20.)*20
            else:
                tempSize = np.ones(len(x))*5
            print tempSize
            predGenreTracking=[]
            #m.plot([tempx[j] for j in range(len(tempx))],[tempy[j] for j in range(len(tempx))], 
            for idx in range(len(tempx)):
                if predGenre[idx] in compareList:
                    regionsToTitle.append(tempReg[idx])
                    h,= m.plot(tempx[idx], tempy[idx],
                            'o', 
                            color=colors[predGenre[idx]], 
                            alpha=0.5,
                            markersize=tempSize[idx],
                            label=genres[i])
                    if predGenre[idx] not in predGenreTracking:
                        predGenreTracking.append(predGenre[idx])
                        labels.append(genres[predGenre[idx]])
                        handles.append(h)

        #add appropriate labels for handles. 
        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color = '#eeefff')
        regions = self.regions
        self.plottedList=[]
        thresh2=904300
        thresh1=304300
        latThresh = 80000
        for item in self.regions:
            if item[0] in regionsToTitle:
                xpt, ypt = m(item[1][0][1],item[1][0][0])
                closest=self.tooClose2([xpt,ypt])
                closestLat, drc =self.tooClose1(ypt)
                mve=90000
                mve2=250000
                if closest > thresh1:
                    if closestLat < latThresh:
                        print 'lat thresh', item[0]
                        print closestLat, 'drc', drc
                        #ax.text(xpt+1000,ypt+drc*mve2,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt+drc*mve2), xytext=None, xycoords='data',textcoords='data') 
                        self.plottedList.append([xpt+1000,ypt+drc*mve2])
                    else:
                        ax.annotate(unicode(item[0]), fontsize=10, xy=(xpt,ypt+mve), xytext=None, xycoords='data',textcoords='data') 
                        #ax.text(xpt+1000,ypt+mve,item[0].encode('ASCII', 'ignore'), fontsize=10) 
                        self.plottedList.append([xpt+1000,ypt+mve])
                #''
                elif closest > thresh2:
                    print item[0]
                    ax.text(xpt+1000,ypt-50000,item[0].encode('ASCII', 'ignore'), fontsize=8) 
                    self.plottedList.append([xpt,ypt])
                #''

        #map.bluemarble()
        # draw the edge of the map projection region (the projection limb)
        m.drawmapboundary()
        # draw lat/lon grid lines every 30 degrees.
        m.drawmeridians(np.arange(0, 360, 30))
        m.drawparallels(np.arange(-90, 90, 30))
        #handles, labels = m.ax.get_legend_handles_labels()
        m.ax.legend(handles,labels);
        '''
        plt.show()
    
    def tooClose1(self,a):
        closest=99999999999999
        drc=1
        for p in self.plottedList:
            d = a-p[1]
            if abs(d) < closest:
                closest = abs(d)
                if d < 0:
                    drc=-1
        return closest, drc 
     
    def tooClose2(self,a):
        closest=99999999999999
        for p in self.plottedList:
            d = sqrt(pow(a[0]-p[0],2)+pow(a[1]-p[1],2))
            if d < closest:
                closest = d
        return closest
        
    def getRegions_artist(self, genreToMap):
        regionDict={}
        for item in self.artistDict.items():
            if genres.index(item[1][1]) in genreToMap: 
                if item[1][0]!='NA':
                    regionDict[item[1][2]]=item[1][0]
        return regionDict

    def getCoords_artist(self):
        lat=[]
        lon=[]
        genre=[]
        for coords in self.artistDict.items():
            if coords[1][0][0]!= 'N':
                lat.append(coords[1][0][0])
                lon.append(coords[1][0][1])
                genre.append(genres.index(coords[1][1]))
        return lat,lon, genre

