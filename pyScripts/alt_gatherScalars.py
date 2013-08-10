from scipy.io.matlab import loadmat
from numpy.random import permutation as P
import numpy as np
import glob
import cPickle
from random import sample
import time as T

def getScalars(foldername="/Users/Tlacael/Documents/MATLAB/LMDdata/LMD60x60_new/", dim=[60,60]):
    files = glob.glob(foldername+"*.mat")
    # This assumes that the onset pattern has shape (N,d0,d1)

    print files[0]
    x = loadmat(files[0])["features"].T.shape
    if x[1] > x[2]:
        d0=x[1]
        d1=x[2]
    else:
        d0=x[2]
        d1=x[1]

    sd = np.array([])
    mu = np.array([])

    count = 0
    
    for i in range(d1):
        print d1
        X = np.zeros((1,d0))
        begT = T.time()
        for f in files:
            x = loadmat(f)
            features = x["features"]
            features = features.T
            fShape = features.shape
            if fShape[0] < 62:
                print fShape
            features =features[sample(range(fShape[0]),fShape[0]),:,:]
            features = features.reshape(fShape[0],fShape[1]*fShape[2])
            X = np.concatenate((X,features[:,i*d0:(i+1)*d0]),axis=0)
            if count%100==0:
                print f
            count=count+1
        print "time remaining: " + str((T.time() - begT)*(d1-(i+1))/60)[:7] + " minutes."

        mu = np.concatenate((mu,X.mean(axis=0)))
        sd = np.concatenate((sd,X.std(axis=0)))
        print str(i+1), "of: ", str(d1)

    mu=mu.reshape(1,len(mu))
    sd=sd.reshape(1,len(sd))
    sd[sd==0]=1
    topDir = "/Users/Tlacael/NYU/RhythmData/"
    outFile =topDir+"LMD_scalars"+ str(dim[0])+"x"+str(dim[1])+".pkl"
    cPickle.dump(np.array([mu,sd]), open(outFile, "wb"))


