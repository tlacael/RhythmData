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
genreNames = ['Axe', 'Bachata', 'Bolero', 
            'Forro', 'Gaucha', 'Merengue', 
            'Pagode', 'Salsa', 'Sertaneja', 'Tango']
 
def gatherData(rootdir): 

    allData = np.array([])
    allGenres = []

    for dirpath, dirnames, filenames in os.walk(rootdir):
        for filename in [f for f in filenames if f.endswith(".pkl")]:
            results = pickle.load(open( os.path.join(dirpath, filename),'r'))
            Inames = results['missedFNames']
            Cnames = results['correctFNames']

            Cg=[int(Cnames[i][0]) for i in range(len(Cnames))]
            Ig=[int(Inames[i][0]) for i in range(len(Inames))]
            genres = np.concatenate((np.array(Cg),np.array(Ig)))
            allGenres = np.concatenate((allGenres,genres))
            allGenres.shape
            data = np.concatenate((np.array(results['confidenceCor']),np.array(results['confidenceInc'])))
            if allData.shape[0] == 0:
                allData = data
            else:
                allData = np.concatenate((allData,data))

    return allData, allGenres

def clusterByGenres(allData, allGenres, visClass=[0,1], genresToShow='all'):

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

