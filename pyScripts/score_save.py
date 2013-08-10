import lmd as L
import collections
import time
import os
import cPickle as pickle

def gather_scores(dsetPath, cur_arch, scalarsPath, dims, validFold=8, fPrefix='', save=False, outfile=''):

    batchSize=10
    
    if not os.path.exists(dsetPath):
        print 'dataset does not exist'
        return
    L.DEFAULT_HDFILE = dsetPath
    print 'loading dataset'
    dset = L.RhythmDataset(scalarsPath, dsetPath, valid=validFold, test=((validFold+1)%10), dim=dims)
    print dset
    print 'building classifier'
    L.arch_lmd = cur_arch

    clf = L.build_clf_model(arch=L.arch_lmd,scale_weights=0.8)
    print 'training'
    
    clf.fit(dset,0.1,batch_size=batchSize,print_freq=1000,EVAL_ALL=False, n_iter=4000)

    print 'validating'
    y_values,fNames, confidenceInc, confidenceCor, correctNames = L.pred_dset_mean(clf,dset,'valid')
    results = {'y_values':y_values,'missedFNames':fNames, 'confidenceInc':confidenceInc, 'confidenceCor':confidenceCor, 'correctFNames':correctNames}
    fNames.sort()
    printScores(cur_arch, y_values, dset, fPrefix, fNames)
    if save:
        #pickle.dump(clf, open(outfile,'w'))        
        newDir = '/Users/Tlacael/NYU/RhythmData/lmd_runs/'+fPrefix[:-6] +'/'
        t = time.localtime()
        curFile = newDir+fPrefix[-5:]+'_'+str(t.tm_hour)+'-'+str(t.tm_min)+'_'+str(t.tm_mday)+'-'+str(t.tm_mon)+'-'+str(t.tm_year)+'_lmd_run_results.pkl'
        pickle.dump(results,open(curFile, 'w' ))
        


def printScores(arch_lmd, y_values, dset, fPrefix, fNames):
    numLayers = len(arch_lmd);
    arch = []
    params = collections.namedtuple('params', 'arch score cMat')
    for i in range(numLayers):#reversed(range(numLayers)):
        layer = arch_lmd[i]
        curLayer = 'layer'+str(i)
        if i==0:
            arch.append([curLayer,layer.weight_shape()])
        else:
            arch.append([curLayer,layer.output_shape()])
    arch = dict(arch)
    score = L.score(y_values)
    cMat = L.confusion_matrix(y_values[:,0],y_values[:,1])
    data = params(arch,score,cMat)
    print 'Architecture: ' , data.arch
    print 'score: ' , data.score
    print 'confusion matrix: '  
    print data.cMat
    print fNames
    
    t = time.localtime()
    newDir = '/Users/Tlacael/NYU/RhythmData/lmd_runs/'+fPrefix[:-6] +'/'
    if not os.path.isdir(newDir):
        os.mkdir(newDir)
    curFile = newDir+fPrefix[-5:]+'_'+str(t.tm_hour)+'-'+str(t.tm_min)+'_'+str(t.tm_mday)+'-'+str(t.tm_mon)+'-'+str(t.tm_year)+'_lmd_run.txt'
    f = open(curFile, 'w' )
    curHDF = dset.hdf.filename
    curHDF = curHDF[-13:-5]
    f.write(curHDF+'\n')
    f.write( 'Architecture: ' + repr(data.arch) + '\n\n' )
    f.write( 'Score: ' + repr(data.score) + '\n\n' )
    f.write( 'Confusion Matrix:\n' + repr(data.cMat) + '\n' )
    f.write( 'incorrectly tagged:\n')
    for i in fNames:
        f.write(i + '\n')
    f.close()

