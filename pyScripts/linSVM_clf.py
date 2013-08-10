import lmd as L
import cPickle as pickle
import h5py
from deeplearn.dataset import Dataset
import numpy as np
from score_save import gather_scores
from sklearn import svm
import lmd as L

#BD or LMD
dsetType = 'lmd'
dsetName = 'lmd'
#### test 128x60 arch
dims = [240,10]
startSize = dims[0]*dims[1]
outputSize = 64

##########################################

default_arch = [L.AffineArgs(weight_shape=(startSize,64)),
        L.SoftmaxArgs(output_shape=(10,))]
topDir = "/Users/Tlacael/NYU/RhythmData/"
dsetP1 = "lmd240x10_6oct.hdf5"
scal1 = "lmd_scalars240x10.pkl"


def buildAndSaveClf(outfile, dset=dsetP1,scalarFile=scal1 , dims=[240,10], arch=default_arch, valid=8):
    
    saveFile = True
    startSize = dims[0]*dims[1]
    print topDir+scalarFile
    gather_scores(topDir+dset,arch,topDir+scalarFile,dims, valid, '', saveFile, outfile)

def transformData(hdfFile, scalarFile, clfFile, validF=8, reuseData=False, dims=[60,25]):
    #architecture without last layer

    #load dataset
    dset = L.RhythmDataset('scalarFiles/'+scalarFile, 'hdfFiles/'+hdfFile, valid=validF, test=(validF+1)%10, dim=dims)
    dsetName = hdfFile[0:2]
    #load pre-computed classifier
    print "loading clf file"
    clf = pickle.load(open(clfFile,'rw')) 
    new_arch = clf.graph.arch[:-1]
    param_values = clf.graph.param_values()
    
    outputSize = new_arch[-1].output_shape()[0]
    new_param_values = {}
    for n in range(len(param_values)-1):
            new_param_values[n] = param_values[n].copy()  # Just to be sure.

    #build with new
    print "building with new arch"
    new_clf = L.build_clf_model(arch=new_arch, scale_weights=0.8)

    new_clf.graph.set_param_values(new_param_values)

    #open or create new hdf5
    new_dset = Dataset('hdfFiles/'+str(dsetName)+str(outputSize)+'_vec.hdf5')
    new_index =[]
    

    print "outputSize ", outputSize

    print "calculating new features"
    for idx in dset.index:
        x=dset.get(idx[0])
        z = np.array([new_clf.fx_output(i) for i in x[0].reshape(x[0].shape[0],1,x[0].shape[1]*x[0].shape[2])])
        cur_idx = new_dset.add_data(X=[z], y=[x[1]],metadata=[{'filename':'NA', 'title':x[2]} ] )
        new_index.append([cur_idx[0],idx[1],idx[2] ])

    #alt for BD

    #must stratify with L.RhythmDataset

    print "closing new dset"
    new_dset.hdf.close() 
    print "reopening to stratify"
    new_dset = L.RhythmDataset(scalarFile, 'hdfFiles/'+str(dsetName)+str(outputSize)+'_vec.hdf5', valid=validF, test=(validF+1)%10, dim=[outputSize,1])
    new_dset.stratify(10)
    print new_dset.folds
    if (reuseData):
        print "copying existing stratification structure"
        new_dset.index = np.array(new_index, np.int32)
        new_dset.split_idx['test']=[[i[0],i[1]] for i in new_dset.index if i[2]==dset.folds['test']]
        new_dset.split_idx['train']=[[i[0],i[1]] for i in new_dset.index if i[2] in dset.folds['train']]
        new_dset.split_idx['valid']=[[i[0],i[1]] for i in new_dset.index if i[2]==dset.folds['valid']]

        new_dset.folds = dset.folds
        print new_dset.folds


'''
#getscalars
xAll=[new_dset.get(i[0])[0] for i in new_dset.index]

xAll = np.concatenate(xAll)

mu = xAll.mean(axis=0)
sd = xAll.std(axis=0)

dim=[1,64]

sd[sd==0]=1
topDir = "/Users/Tlacael/NYU/RhythmData/"
outFile =topDir+dsetType+"_scalars"+ str(dim[0])+"x"+str(dim[1])+".pkl"
cPickle.dump(np.array([mu,sd]), open(outFile, "wb"))
'''

#SVM


def linSVM(new_dset, validF):
    print "loading dataset"
    new_dset = L.RhythmDataset('/Users/Tlacael/NYU/RhythmData/lmd_scalars1x64.pkl',"/Users/Tlacael/NYU/RhythmData/"+new_dset,valid=validF,test=(validF+1)%10, dim=[64,1])
    #get training set
    print "loading training set"
    xAll = [new_dset.get(i[0])[0] for i in new_dset.split_idx['train']]
    xAll = np.concatenate(xAll)
    xAll = xAll.reshape(xAll.shape[0],xAll.shape[2])
    
    #get classes for training set
    print "loading validation set"
    classAll=[np.tile(new_dset.get(i[0])[1],(new_dset.get(i[0])[0].shape[0],)) for i in new_dset.split_idx['train']]
    target=np.concatenate(classAll)

    #get validation set
    xVerify = [new_dset.get(i[0])[0] for i in new_dset.split_idx['valid']]
    xVerify = np.concatenate(xVerify)
    xVerify = xVerify.reshape(xVerify.shape[0],xVerify.shape[2])

    
    classVer=[np.tile(new_dset.get(i[0])[1],(new_dset.get(i[0])[0].shape[0],)) for i in new_dset.split_idx['valid']]
    targetVer=np.concatenate(classVer)


    print "building model"
    svc = svm.SVC(kernel='linear', verbose=True)
    print "fit data"
    svc.fit(xAll,target)

    scre = svc.score(xVerify,targetVer)
    print "score: ", scre
    return scre









