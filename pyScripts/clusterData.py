import numpy as np
from pyScripts import lmd as L
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from shutil import copyfile
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import os
import os.path
import cPickle as pickle
from mpl_toolkits.mplot3d import Axes3D
from Levenshtein import ratio
import eyed3 as d3
import cPickle as pickle
from operator import itemgetter, attrgetter
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import figure, show
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from subprocess import Popen, PIPE
import signal
from time import time

colors = ['#00ffff', '#000000', '#0000ff', 
                '#ffffff', '#7fff00', '#ff8c00', 
                '#eedd82', '#ff00ff', '#ff0000', '#ffff00']
      
genreNames = ['Axe', 'Bachata', 'Bolero', 
            'Forro', 'Gaucha', 'Merengue', 
            'Pagode', 'Salsa', 'Sertaneja', 'Tango']
ctemp= plt.get_cmap('spectral', 10)
for i in range(10):
    colors[i]=ctemp(i)

class ClusterLMD:
    def __init__(self, folder, save=False):

        if save:
            clus = pickle.load(open(u'/Users/Tlacael/NYU/RhythmData/pyScripts/clus.pkl', 'r'))
            self.allData       = clus.allData 
            self.allGenres     = clus.allGenres
            self.allPredGenres = clus.allPredGenres
            self.allSongs      = clus.allSongs
            self.filePaths     = clus.filePaths
            self.artists       = clus.artists
            self.regions       = clus.regions
            self.bpm           = clus.bpm


        else:
            self.allData, self.allGenres, self.allPredGenres, self.allSongs = self.gatherData(folder)
            self.filePaths = [self.findID3tagLocation(self.allSongs[i], self.allGenres[i]) for i in range(len(self.allSongs))]
            self.artists = self.getSongInfoFromFiles()
            self.regions = self.getRegions()
            self.allData = np.array(self.allData)
            self.bpm = self.getBPM()


    def getBPM(self):
        tempoDict = pickle.load(open(u'/Users/Tlacael/NYU/RhythmData/pyScripts/tempoDict.pkl', 'r'))
        bpm=[]
        count = 0
        for i in range(len(self.filePaths)):
            try:
                bpm.append(tempoDict[self.filePaths[i]])
                count +=1
            except:
                print self.filePaths[i]
                bpm.append('NA')

        print count
        return bpm

    def barGraphByRegion(self):
        '''
        graph shows difference between percentage of songs by state and 
        percentage of misses by state.
        '''
        songPerRegion={}
        missPerRegion={}
        totMiss = 0
        percentReg= {} 
        genrePerRegion={}
        

        #initialize
        for r in self.regions:
            songPerRegion[r[0].split(' - ')[-1]]=0
            missPerRegion[r[0].split(' - ')[-1]]=0
            genrePerRegion[r[0].split(' - ')[-1]]={}
        for r in self.regions:
            for i in range(10):
                genrePerRegion[r[0].split(' - ')[-1]][i]=0
        #get num songs per region
        for r in self.regions:
            songPerRegion[r[0].split(' - ')[-1]]+=1
        #get number of missed per region
        for i in range(len(self.allGenres)):
            if self.allGenres[i] != self.allPredGenres[i]:
                missPerRegion[self.regions[i][0].split(' - ')[-1]]+=1
                totMiss+=1
        #get number of genres per region
        for i in range(len(self.allGenres)):
            genrePerRegion[self.regions[i][0].split(' - ')[-1]][int(self.allGenres[i])]+=1
        
        #get percentage score difference
        for i in range(len(missPerRegion)):
            percentReg[songPerRegion.keys()[i]]=(songPerRegion.values()[i]/float(len(self.regions)) - \
                    missPerRegion.values()[i]/float(totMiss))*100;

        
        def autolabel(rects):
        # attach some text labels
            for rect in rects:
                height = rect.get_height()
                if rect.get_y() < 0: 
                    height=0.0
                ax.text(rect.get_x()+rect.get_width()/2., 
                    0.05+height, percentReg[rect.get_x()][0], 
                    ha='center', va='bottom', rotation='vertical', size='small')
       

 
        def autolabel2(rects, prev):
        # attach some text labels
            n=0
            for rect in rects:
                height = rect.get_height()
                if rect.get_y() < 0: 
                    height=0.0
                ax.text(rect.get_x()+rect.get_width()/2., 
                    2+prev[n], genrePerRegion[rect.get_x()][0], 
                    ha='center', va='bottom', rotation='vertical', size='small')
                n+=1

        percentReg.pop('NA')
        percentReg = sorted(percentReg.items(),key=lambda x: x[0])
        N = len(percentReg)
        fig, ax = plt.subplots()
        rects1 = ax.bar(np.arange(N), [i[1] for i in percentReg])
        autolabel(rects1)
       
        self.pr = percentReg
        fig, ax = plt.subplots()
        self.gpr = genrePerRegion
        genrePerRegion.pop('NA')
        genrePerRegion = sorted(genrePerRegion.items(),key=lambda x: x[0])
        data = [i[1].values() for i in genrePerRegion]
        data = np.array(data).T
        N2 = data.shape[1]
        prev = np.zeros((N2,))
        g = 0
        for d in data:
            d = d.reshape(len(d))
            rects2 = ax.bar(np.arange(N2), d, bottom=prev, color=colors[g], label=genreNames[g])
            g+=1
            prev = data[0:g].sum(axis=0)
        autolabel2(rects2, prev)
        
        handles, labels = ax.get_legend_handles_labels()

        #shrink text
        fontP = FontProperties()
        fontP.set_size('small')
        #shrink box to fit legend on left
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(handles, labels,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
            prop = fontP);

        plt.show()
            

    def getRegions(self):
        artistDictFile = '/Users/Tlacael/NYU/RhythmData/pyScripts/artistDict.pkl'
        regions=[]
        f = open(artistDictFile, 'r')
        artistDict = pickle.load(f)
        for item in self.artists:
            found = 0
            for key in artistDict:
                if ratio(item, key) > 0.9:
                    found +=1
                    art = key
            regions.append([artistDict[art][1], artistDict[art][2]])
        return regions


 
    def getSongInfoFromFiles(self):
        allTags=[d3.load(name).tag for name in self.filePaths]
        artists=[]
        dup=2;
        for tag in allTags:
            artists.append(tag.artist)

        return artists
        

    def gatherData(self,rootdir): 

        allData = []
        allGenres = []
        allPredGenres = []
        allSongs=[]

        for dirpath, dirnames, filenames in os.walk(rootdir):
            for filename in [f for f in filenames if f.endswith(".pkl")]:
                results = pickle.load(open( os.path.join(dirpath, filename),'r'))
                    
                for item in results:
                    allData.append(np.array(item[0]))
                    allSongs.append(item[1])
                    allGenres.append(item[2])
                    allPredGenres.append(item[3])

        return allData, allGenres, allPredGenres, allSongs


    def findID3tagLocation(self, songTitle, genre):
        LMDdir='/Users/Tlacael/NYU/Latin_Music_Data/LMD/'
        maxRatio=0
        path=[]
        for r,d,f in os.walk(LMDdir+genreNames[int(genre)]):
            for files in f:
                if files.endswith('.mp3'):
                    temp = ratio(songTitle, files)
                    if temp > maxRatio:
                        maxRatio=temp
                        path = os.path.join(r,files)
        return path
 
    def mapLatLongToColor(self, lat,lng, maxlat, minlat, maxlng, minlng):
        if lat =='N':
            return [0,0,0]
        lat = (lat - minlat)/(maxlat - minlat)
        lng = (lng - minlng)/(maxlng - minlng)
        return [lat,lng, 1-lat]



    def CompClusterByGenresRegion(self,visClass=[0,1], genresToShow=[2]):

        allData = self.allData
        allGenres = self.allGenres
        allPredGenres = self.allPredGenres
        regions = self.regions

        dims = len(visClass)
        plt.close()
        fig = plt.figure()
        if dims ==3:
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(111)

        ph = []
        if genresToShow != 'all':
            classRange = genresToShow
        else:
            classRange = range(10)
        #find max for color map
        maxlat=-90
        minlat=90
        maxlng=-180
        minlng=180
        for j in classRange:
             if dims == 2:           
                 for i in range(len(allData[:,visClass[0]])): 
                     if allGenres[i]==j and regions[i][0] != 'Arizona, USA':
                         if regions[i][1][0]!='N':
                             if regions[i][1][0] > maxlat:
                                 maxlat = regions[i][1][0]
                             if regions[i][1][0] < minlat:
                                 minlat = regions[i][1][0]
                             if regions[i][1][1] > maxlng:
                                 maxlng = regions[i][1][1]
                             if regions[i][1][1] < minlng:
                                 minlng = regions[i][1][1]
        print maxlat, minlat, maxlng, minlng
        #create handles for genre marker shape legend
        h1, = ax.plot(0,0, 'o', c='w', label=genreNames[classRange[0]])
        regionDict={}
        infoLookup={}
        for j in classRange:
            if dims == 2:
                for i in range(len(allData[:,visClass[0]])): 
                    try: 
                        print infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]
                        print i
                        
                    except:
                        infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]=i
                        print 'no collision'

                    if allGenres[i]==j and allPredGenres[i] in visClass and regions[i][0] != 'Arizona, USA':# and allGenres[i] != allPredGenres[i]:
                        color = self.mapLatLongToColor(regions[i][1][0],regions[i][1][1], maxlat, minlat, maxlng, minlng)
                        #make each genre circel or square  
                        regionDict[regions[i][0]] = abs(color[0]) + abs(color[1]) + abs(color[2])
                        ax.plot( 
                            allData[i,visClass[0]], 
                            allData[i,visClass[1]], 
                            'o', c=color, label=regions[i][0], picker=5)
                               
            elif dims ==3:
                 ph.append(ax.plot( 
                    [allData[i,visClass[0]] for i in range(len(allData[:,visClass[0]])) if allGenres[i]==j], 
                    [allData[i,visClass[1]] for i in range(len(allData[:,visClass[1]])) if allGenres[i]==j], 
                    [allData[i,visClass[2]] for i in range(len(allData[:,visClass[2]])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
        
        def onpick3(event):
            try:
                self.p.send_signal(signal.SIGINT)
                self.p.wait()
            except:
                pass
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            idx = infoLookup[np.take(xdata, ind)[0]+np.take(ydata, ind)[0]]
            tag = d3.load(self.filePaths[idx]).tag 
            for i in range(10):
                print genreNames[i],round(allData[idx][i], 3)
            print 'Artist:', tag.artist
            print 'Song:', tag.title
            print 'Genre:', genreNames[int(allGenres[idx])]
            print 'Predicted Genre:', genreNames[int(allPredGenres[idx])]
            print 'Region:', regions[idx][0]
            print 'File path:', self.filePaths[idx]
            self.p = Popen(["play", self.filePaths[idx]], 
                    stdout=PIPE)
            #print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        fig.canvas.mpl_connect('pick_event', onpick3)
        #get handles
        regionHandles={}
        handles, labels = ax.get_legend_handles_labels()
        #remove redundant handles
        for i in range(len(handles)):
            regionHandles[labels[i]] = handles[i]
        
        #force order
        regionTuple = regionDict.items()
        regionTuple = sorted(regionTuple, key=itemgetter(1))

        handles = [regionHandles[i[0]] for i in regionTuple]
        handles.append(h1)
        labels  = [i[0] for i in regionTuple]
        labels.append(genreNames[classRange[0]])
        
        #shrink text
        fontP = FontProperties()
        fontP.set_size('small')
        #shrink box to fit legend on left
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(handles, labels,
                loc='center left', 
                bbox_to_anchor=(1, 0.5), 
                fancybox=True, 
                shadow=True,
                prop = fontP);

        ax.set_xlabel(genreNames[visClass[0]])
        ax.set_ylabel(genreNames[visClass[1]])
        if dims ==3:
            ax.set_zlabel(genreNames[visClass[2]])
        plt.show()
           
        try:
            print "killing process"
            self.p.send_signal(signal.SIGINT)
            self.p.wait()
        except:
            pass
    
    def mapBPMToColor(self, bpm, maxBPM, minBPM):


        val = (bpm - minBPM)/float(maxBPM - minBPM)
        if val - 0.5 < 0:
            return (val,1, 1)
        else:
            return (1,1-val, 1)



    def clusterByGenresTempos(self,visClass=[0,1], genresToShow='all', tempoPlot=False):

        allData = self.allData
        allGenres = self.allGenres
        allPredGenres = self.allPredGenres
        regions = self.regions

        dims = len(visClass)
        plt.close()
        fig = plt.figure()
        if dims ==3:
            ax = fig.add_subplot(121,projection='3d')
        else:
            ax = fig.add_subplot(111)
            axc = fig.add_axes([0.05, 0.80, 0.9, 0.15])

        ph = []
        if genresToShow != 'all':
            classRange = genresToShow
        else:
            classRange = range(10)
         
        maxBPM = max([self.bpm[i] for i in range(len(self.bpm)) if self.bpm[i]!="NA" and allGenres[i] in classRange])
        minBPM = min([self.bpm[i] for i in range(len(self.bpm)) if self.bpm[i]!="NA" and self.bpm[i]!=0 and allGenres[i] in classRange])
        print maxBPM, minBPM
        #create handles for genre marker shape legend
        h1, = ax.plot(0,0, 'o', c='w', label=genreNames[classRange[0]])
        if len(classRange)>1:
            print len(classRange)
            h2, = ax.plot(0,0, 's', c='w', label=genreNames[classRange[1]])
        regionDict={}
        infoLookup={}
        cmap= plt.get_cmap('hot', int(maxBPM-minBPM))
        colorList = cmap(range(int(maxBPM-minBPM)))

        for i in range(len(regions)):
            try:
                print self.allSongs[infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]], 'old'
                print self.allSongs[i], 'new'
            except:
                infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]=i
                #print 'no collisions'
        for j in classRange:
            if dims == 2:
                for i in range(len(allData[:,visClass[0]])): 
                    if allGenres[i]==j and self.bpm[i]!='NA' and self.bpm[i]!=0:
                        #make each genre circle or square  
                        if j == classRange[0] and allPredGenres[i] in classRange:
                            color = colorList[int(self.bpm[i]-minBPM)-1]
                            regionDict[regions[i][0]] = abs(color[0]) + abs(color[1]) + abs(color[2])
                            ax.plot( 
                                allData[i,visClass[0]], 
                                allData[i,visClass[1]], 
                                'o', c=color, label=regions[i][0], picker=5)
                        if len(classRange)>1:
                            if j == classRange[1] and allPredGenres[i] in classRange:
                                color = colorList[int(self.bpm[i]-minBPM)-1]
                                regionDict[regions[i][0]] = abs(color[0]) + abs(color[1]) + abs(color[2])
                                ax.plot( 
                                    allData[i,visClass[0]], 
                                    allData[i,visClass[1]], 
                                    's', c=color, label=regions[i][0], picker=5)

         
        def onpick3(event):
            if time() - timeElapsed < 1.:
                return
            try:
                self.p.send_signal(signal.SIGINT)
                self.p.wait()
            except:
                pass
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            idx = infoLookup[np.take(xdata, ind)[0]+np.take(ydata, ind)[0]]
            self.wholeLib=infoLookup
            tag = d3.load(self.filePaths[idx]).tag 
            for i in range(10):
                print genreNames[i],round(allData[idx][i], 3)
            print 'Artist:', tag.artist
            print 'Song:', tag.title
            print 'Genre:', genreNames[int(allGenres[idx])]
            print 'Predicted Genre:', genreNames[int(allPredGenres[idx])]
            print 'Region:', regions[idx][0]
            print 'Tempo: ', self.bpm[idx]
            print 'File path:', self.filePaths[idx]
            self.p = Popen(["play", self.filePaths[idx]], 
                    stdout=PIPE)
            #print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        
        timeElapsed = 0
        fig.canvas.mpl_connect('pick_event', onpick3)
        #get handles
        regionHandles={}
        handles, labels = ax.get_legend_handles_labels()
        #remove redundant handles
        for i in range(len(handles)):
            regionHandles[labels[i]] = handles[i]
        
        #force order
        regionTuple = regionDict.items()
        regionTuple = sorted(regionTuple, key=itemgetter(1))

        handles = [regionHandles[i[0]] for i in regionTuple]
        handles.append(h1)
        if len(genresToShow) >1:
            handles.append(h2)
        labels  = [i[0] for i in regionTuple]
        labels.append(genreNames[classRange[0]])

        if len(classRange)>1:
            labels.append(genreNames[classRange[1]])
        
        #shrink text
        fontP = FontProperties()
        fontP.set_size('small')
        #shrink box to fit legend on left
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width , box.height*0.8])
        
        boxc = axc.get_position()
        axc.set_position([box.x0, boxc.y0, box.width , boxc.height*0.2])
        axc.set_xlabel("BPM")

        ax.set_xlabel(genreNames[visClass[0]])
        ax.set_ylabel(genreNames[visClass[1]])
        if dims ==3:
            ax.set_zlabel(genreNames[visClass[2]])
        norm = mpl.colors.Normalize(vmin=minBPM, vmax=maxBPM)

        cb1 = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                norm=norm,         
                label="BPM",
                orientation='horizontal')
        #axc.add_axes([0.05, 0.80, 0.9, 0.15])
        rect = fig.patch
        rect.set_facecolor('white')

        
        plt.show()
           
        try:
            print "killing process"
            self.p.send_signal(signal.SIGINT)
            self.p.wait()
        except:
            pass

 
    def clusterByGenresRegion(self,visClass=[0,1], genresToShow='all', tempoPlot=False):

        allData = self.allData
        allGenres = self.allGenres
        allPredGenres = self.allPredGenres
        regions = self.regions

        dims = len(visClass)
        plt.close()
        fig = plt.figure()
        if dims ==3:
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(111)

        ph = []
        if genresToShow != 'all':
            classRange = genresToShow
        else:
            classRange = range(10)
        #find max for color map
        maxlat=-90
        minlat=90
        maxlng=-180
        minlng=180
        for j in classRange:
             if dims == 2:           
                 for i in range(len(allData[:,visClass[0]])): 
                     if allGenres[i]==j and regions[i][0] != 'Arizona, USA':
                         if regions[i][1][0]!='N':
                             if regions[i][1][0] > maxlat:
                                 maxlat = regions[i][1][0]
                             if regions[i][1][0] < minlat:
                                 minlat = regions[i][1][0]
                             if regions[i][1][1] > maxlng:
                                 maxlng = regions[i][1][1]
                             if regions[i][1][1] < minlng:
                                 minlng = regions[i][1][1]
        print maxlat, minlat, maxlng, minlng
        #create handles for genre marker shape legend
        h1, = ax.plot(0,0, 'o', c='w', label=genreNames[classRange[0]])
        h2, = ax.plot(0,0, 's', c='w', label=genreNames[classRange[1]])
        regionDict={}
        infoLookup={}
        for i in range(len(regions)):
            try:
                print self.allSongs[infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]], 'old'
                print self.allSongs[i], 'new'
            except:
                infoLookup[allData[i,visClass[0]]+allData[i,visClass[1]]]=i
                #print 'no collisions'
        for j in classRange:
            if dims == 2:
                for i in range(len(allData[:,visClass[0]])): 
                    if allGenres[i]==j and regions[i][0] != 'Arizona, USA':# and allGenres[i] != allPredGenres[i]:
                        color = self.mapLatLongToColor(regions[i][1][0],regions[i][1][1], maxlat, minlat, maxlng, minlng)
                        #make each genre circle or square  
                        if j == classRange[0] and allPredGenres[i] in classRange:
                            regionDict[regions[i][0]] = abs(color[0]) + abs(color[1]) + abs(color[2])
                            ax.plot( 
                                allData[i,visClass[0]], 
                                allData[i,visClass[1]], 
                                'o', c=color, label=regions[i][0], picker=5)
                        if j == classRange[1] and allPredGenres[i] in classRange:
                            regionDict[regions[i][0]] = abs(color[0]) + abs(color[1]) + abs(color[2])
                            ax.plot( 
                                allData[i,visClass[0]], 
                                allData[i,visClass[1]], 
                                's', c=color, label=regions[i][0], picker=5)

        
            elif dims ==3:
                 ph.append(ax.plot( 
                    [allData[i,visClass[0]] for i in range(len(allData[:,visClass[0]])) if allGenres[i]==j], 
                    [allData[i,visClass[1]] for i in range(len(allData[:,visClass[1]])) if allGenres[i]==j], 
                    [allData[i,visClass[2]] for i in range(len(allData[:,visClass[2]])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
         
        def onpick3(event):
            if time() - timeElapsed < 1.:
                return
            try:
                self.p.send_signal(signal.SIGINT)
                self.p.wait()
            except:
                pass
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            idx = infoLookup[np.take(xdata, ind)[0]+np.take(ydata, ind)[0]]
            tag = d3.load(self.filePaths[idx]).tag 
            for i in range(10):
                print genreNames[i],round(allData[idx][i], 3)
            print 'Artist:', tag.artist
            print 'Song:', tag.title
            print 'Genre:', genreNames[int(allGenres[idx])]
            print 'Predicted Genre:', genreNames[int(allPredGenres[idx])]
            print 'Region:', regions[idx][0]
            print 'BPM: ', self.bpm[idx]
            print 'File path:', self.filePaths[idx]
            self.p = Popen(["play", self.filePaths[idx]], 
                    stdout=PIPE)
            #print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        
        timeElapsed = 0
        fig.canvas.mpl_connect('pick_event', onpick3)
        #get handles
        regionHandles={}
        handles, labels = ax.get_legend_handles_labels()
        #remove redundant handles
        for i in range(len(handles)):
            regionHandles[labels[i]] = handles[i]
        
        #force order
        regionTuple = regionDict.items()
        regionTuple = sorted(regionTuple, key=itemgetter(1))

        handles = [regionHandles[i[0]] for i in regionTuple]
        handles.append(h1)
        handles.append(h2)
        labels  = [i[0] for i in regionTuple]
        labels.append(genreNames[classRange[0]])
        labels.append(genreNames[classRange[1]])
        
        #shrink text
        fontP = FontProperties()
        fontP.set_size('small')
        #shrink box to fit legend on left
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        
        ax.legend(handles, labels,
                loc='center left', 
                bbox_to_anchor=(1, 0.5), 
                fancybox=True, 
                shadow=True,
                prop = fontP);

        ax.set_xlabel(genreNames[visClass[0]])
        ax.set_ylabel(genreNames[visClass[1]])
        if dims ==3:
            ax.set_zlabel(genreNames[visClass[2]])
        plt.show()
           
        try:
            print "killing process"
            self.p.send_signal(signal.SIGINT)
            self.p.wait()
        except:
            pass

       

    def clusterByGenres(self, visClass=[0,1], genresToShow='all'):

        allData = self.allData
        allGenres = self.allGenres
        allPredGenres = self.allPredGenres

        dims = len(visClass)
        colors = ['#00ffff', '#000000', '#0000ff', 
                '#ffffff', '#7fff00', '#ff8c00', 
                '#eedd82', '#ff00ff', '#ff0000', '#ffff00']
        plt.close()
        fig = plt.figure()
        if dims ==3:
            ax = fig.add_subplot(111,projection='3d')
        else:
            ax = fig.add_subplot(111)

        ph = []
        if genresToShow != 'all':
            classRange = genresToShow
        else:
            classRange = range(10)

        for j in classRange:
            if dims == 2:
                ph.append(ax.plot( 
                    [allData[i,visClass[0]] for i in range(len(allData[:,visClass[0]])) if allGenres[i]==j], 
                    [allData[i,visClass[1]] for i in range(len(allData[:,visClass[1]])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
        
            elif dims ==3:
                 ph.append(ax.plot( 
                    [allData[i,visClass[0]] for i in range(len(allData[:,visClass[0]])) if allGenres[i]==j], 
                    [allData[i,visClass[1]] for i in range(len(allData[:,visClass[1]])) if allGenres[i]==j], 
                    [allData[i,visClass[2]] for i in range(len(allData[:,visClass[2]])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
        

        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels);
        ax.set_xlabel(genreNames[visClass[0]])
        ax.set_ylabel(genreNames[visClass[1]])
        if dims ==3:
            ax.set_zlabel(genreNames[visClass[2]])
        plt.show()
            

     
      
    def clusterDataPCA(allData, allGenres, visClass='all', dims=2):

        clf = PCA(n_components=dims)
        pcaX_true = clf.fit_transform(allData)

        colors = ['#00ffff', '#000000', '#0000ff', 
                '#ffffff', '#7fff00', '#ff8c00', 
                '#eedd82', '#ff00ff', '#ff0000', '#ffff00']
        plt.close()
        fig = plt.figure()
        ax = Axes3D(fig)
        ph = []
        if visClass != 'all':
            classRange = visClass
        else:
            classRange = range(10)

        for j in classRange:
            if dims == 2:
                ph.append(ax.plot( 
                    [pcaX_true[i,0] for i in range(len(pcaX_true[:,0])) if allGenres[i]==j], 
                    [pcaX_true[i,1] for i in range(len(pcaX_true[:,1])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
        
            elif dims ==3:
                 ph.append(ax.plot( 
                    [pcaX_true[i,0] for i in range(len(pcaX_true[:,0])) if allGenres[i]==j], 
                    [pcaX_true[i,1] for i in range(len(pcaX_true[:,1])) if allGenres[i]==j], 
                    [pcaX_true[i,2] for i in range(len(pcaX_true[:,2])) if allGenres[i]==j], 
                    'o', c=colors[j], label=genreNames[j]));
        

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels);
        plt.show()
            

      

    def getAllDataFromHdf(dset, dims):
            print "loading dataset"
                
            new_dset = L.RhythmDataset('none',"/Users/Tlacael/NYU/RhythmData/hdfFiles/"+dset, dim=dims)
            print new_dset.hdf.file
            #get training set
            print "loading training set"

            allData = [new_dset.get(i[0])[0].reshape(new_dset.get(i[0])[0].shape[0],dims[0]*dims[1]).mean(axis=0) for i in new_dset.index]
            allData = np.array(allData)

            #get classes for training set
            print "loading validation set"
            classAll=[i[1] for i in new_dset.index]
            classAll = np.array(classAll)
            return allData, classAll

    def copyMP3sByPrediction(resultsFolder):

        mp3sFolder = resultsFolder + '/mp3s/'
        if not os.path.isdir(mp3sFolder):
            os.mkdir(mp3sFolder)
        
        for dirpath, dirnames, filenames in os.walk(resultsFolder):
            for filename in [f for f in filenames if f.endswith(".pkl")]:
                results = pickle.load(open( os.path.join(dirpath, filename),'r'))

                savePath = mp3sFolder + 'allMP3s'+'/'
                if not os.path.isdir(savePath):
                    os.mkdir(savePath)
                
                Inames = results['missedFNames']
                Cnames = results['correctFNames']
                songTitles = np.concatenate((Cnames,Inames))
                
                for t in songTitles:
                    print t
                    if not os.path.isdir(savePath+genreNames[int(t[0])]):
                        os.mkdir(savePath+genreNames[int(t[0])])
                    
                    finDest = savePath+genreNames[int(t[0])]+'/'+genreNames[int(t[2])] + '/'
                    if not os.path.isdir(finDest):
                        os.mkdir(finDest)
                    pth = os.path.abspath('/Users/Tlacael/Documents/MATLAB/LMD-mp3-mid/'+genreNames[int(t[0])]+'/'+t[4:])
                    copyfile(pth, finDest+t[4:])

