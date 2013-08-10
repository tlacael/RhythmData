'''
Created on Mar 10, 2013

@author: ejhumphrey
'''

import os, glob
import cPickle

import numpy as np
from scipy.io.matlab import loadmat
from sklearn.metrics import confusion_matrix

from deeplearn.base import randint, randitem
from deeplearn.core.layers import AffineArgs, SoftmaxArgs
from deeplearn.core.graphs import Network
from deeplearn.core.modules import Monitor
from deeplearn.dataset import Dataset, kTRAIN
from deeplearn.models.supervised import Likelihood

# Change these.
BASE_PATH = "/Users/Tlacael/NYU/RhythmData "
DEFAULT_HDFILE = "/Users/Tlacael/NYU/RhythmData/jpb_lmd.hdf5"
DEFAULT_HDFILE = "/Users/Tlacael/NYU/RhythmData/lmd96x120.hdf5"
LMD_SCALAR = "bypass"#"/Users/Tlacael/NYU/RhythmData/lmd_scalars.pkl"


GENRES = {}

def collect_mat_files(dpath,dname='LMD',didx=0):
    """
    convenience function to collect filepaths to precomputed 
    mat-files.
    
    Parameters
    ----------
    dpath : str
        filepath to a dataset of .mat files
    didx:
        [0:LMD, 1:Ballroom]
    
    Returns
    -------
    files : list
        string file paths
    
    """
    #dnames = ['LMD','Ballroom']
    #dname = dnames[didx]
    return glob.glob(os.path.join(dpath, dname,'*.mat'))

def import_data(hdfile, didx=0, dpath=BASE_PATH, PASS=False, dname='LMD'):
    """
    collects precomputed mat-files and load the data into a 
    Dataset object.
    
    Parameters
    ----------
    hdfile : str
        filepath to save Dataset object
    dpath : str
        filepath to a dataset of .mat files
    didx: int, default 0
        [0:LMD, 1:Ballroom]
    PASS : bool, default=False
        for testing, don't actually add anything to the dataset
        
    Returns
    -------
    genres : dict
        key-value pairs of genre tag (str) and class (int)
    """
    dset = Dataset(hdfile)
    
    genres = {}
    for f in collect_mat_files(dpath,dname, didx):
        data = loadmat(f)
        g = data['genre'][0]
        print f
        if not g in genres:
            genres[g] = len(genres)
        
        if not PASS:
            x = data['features'].T
            dset.add_data(X=[x],y=[genres[g]],
                          metadata=[{'filename':os.path.split(f)[-1]}])
        
    return genres


class RhythmDataset(Dataset):
    
    def __init__(self, hdfile=DEFAULT_HDFILE, valid=None, test=None):
        Dataset.__init__(self, hdfile, valid=valid, test=test)
        self.scalars = []
        if os.path.exists(LMD_SCALAR):
            mu,sd = cPickle.load(open(LMD_SCALAR,'r'))
            self.scalars += [mu,sd]
            print "scalars found" 

    def training_batch(self, batch_size):
        assert len(self.folds[kTRAIN])
        return self._uniform_batch(batch_size, fold_set=kTRAIN)
    
    def get(self, ikey):
        x,y = Dataset.get(self, ikey)
        if len(self.scalars):
            x = (x - self.scalars[0].reshape(1,120,96))/self.scalars[1].reshape(1,120,96)
        return x, y
    
    def sample(self, ikey):
        x,y = self.get(ikey)
        # pick a single observation of x
        return randitem(x),y
    
    def reshape_batch(self, x):
        x = np.asarray(x)
        shp = x.shape
        return x.reshape(shp[0], np.prod(shp[1:]))
        
        
class RhythmLikelihood(Likelihood):
    def __init__(self, graph, **kwargs):
        Likelihood.__init__(self, graph, **kwargs)
        self.best_valid_error = np.inf
        self.best_param_values = None
        
    def fit(self, dset, learning_rate, n_iter=5000,
            batch_size=100, print_freq=25, EVAL_ALL=True):
        
        """
        Trains the model... safely catches keyboard interrupts
        without losing progress.
        
        Parameters
        ----------
        dset : Dataset
            pre-loaded with data and whatnot
        learning_rate : scalar
            update rate for each iteration
        n_iter : int
            maximum iterations to run
        batch_size : int
            number of datapoints to fetch per update
        print_freq : int
            number of iterations between displaying stats
        EVAL_ALL : bool
            compute validation over entire split
            
        Returns
        -------
        awesomeness : float
        
        ...I'm kidding, it returns None
        """
        
        learning_rate = float(learning_rate) # Just in case...
        
        if self.mntr is None:
            self.mntr = Monitor(max_iter=n_iter)
        
        self.mntr.reset_clock()
        
        while not self.mntr.finished():
            try:
                x,y = dset.training_batch(batch_size)
                l = self.fx_update(x,y,learning_rate)
                e = np.mean(self.fx_error(x,y)*100)
                
                self.mntr.update(loss=l, error=e)
                if (self.mntr.iter()%print_freq) == 0:
                    if EVAL_ALL:
                        valid_error = 100 - score(pred_dset_max(self.fx_output, dset, 'train'))
                        if valid_error < self.best_valid_error:
                            self.best_param_values = self.graph.param_values()
                            self.best_valid_error = valid_error 
                        self.mntr.stats['Validation Error'] = "%0.3f"%valid_error
                    else:
                        self.mntr.stats['Validation Error'] = "N/A"
                    print self.mntr
            except KeyboardInterrupt:
                break
        print "\nFinished\n%s"%self.mntr
        
        
def parse_dset(clf, dset, idx):
    """
    given a likelihood classifier, compute the probability surfaces of a
    set of datapoints
    
    Parameters
    ----------
    clf : RhythmLikelihood object
        trained classifier
    dset: RhythmDataset object
        rhythm dataset 
    idx : np.ndarray
        collection of absolute integer keys into the dset
        
    Returns
    -------
    probs : list
        probability surfaces over the selected data
    """
    probs = []
    for s in range(int(np.ceil(len(idx)/100.0))):
        lower = s*100
        upper = min([(s+1)*100,len(idx)])
        X = np.array([dset.get(idx[k])[0] for k in range(lower,upper)])
        probs += [clf.fx_output(dset.reshape_batch(X))] 
    
    return probs

def pred_dset_mean(clf, dset, fold_set):
    """
    given a likelihood classifier, predict the class of each datapoint 
    by taking the mean over each observation
    
    Parameters
    ----------
    clf : RhythmLikelihood object
        trained classifier
    dset: RhythmDataset object
        rhythm dataset 
    fold_set : str
        one of ['train','valid','test']
        
    Returns
    -------
    y_values : 2d np.ndarray
        actual and predicted classes for each datapoint in the
        fold set.
    """
    y_values = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        p = clf.fx_output(dset.reshape_batch(x))
        y_pred = np.mean(p,axis=0).argmax()
        y_values += [(y, y_pred)]
    
    return np.asarray(y_values)     
        
def pred_dset_weight_vote(fx_output, dset, fold_set):
    """
    given a likelihood classifier, predict the class of each datapoint 
    by taking the max over each observation and performing weighted voting
    
    Parameters
    ----------
    clf : RhythmLikelihood object
        trained classifier
    dset: RhythmDataset object
        rhythm dataset 
    fold_set : str
        one of ['train','valid','test']
        
    Returns
    -------
    y_values : 2d np.ndarray
        actual and predicted classes for each datapoint in the
        fold set.
    """
    y_values = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        p = fx_output(dset.reshape_batch(x))
        y_pred = p.argmax(axis=1)
        y_weight = p.max(axis=1)
        h = np.bincount(y_pred,weights=y_weight)
        y_pred = h.argmax()
        y_values += [(y, y_pred)]
    
    return np.asarray(y_values)

def pred_dset_max(fx_output, dset, fold_set):
    """
    given a likelihood classifier, predict the class of each datapoint 
    by taking the max over the entire probability surface
    
    Parameters
    ----------
    clf : RhythmLikelihood object
        trained classifier
    dset: RhythmDataset object
        rhythm dataset 
    fold_set : str
        one of ['train','valid','test']
        
    Returns
    -------
    y_values : 2d np.ndarray
        actual and predicted classes for each datapoint in the
        fold set.
    """
    labels = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        p = fx_output(dset.reshape_batch(x))
        row_max = p.argmax(axis=1)
        col_max = p[np.arange(len(p)),row_max].argmax()
        y_pred = row_max[col_max]
        labels += [(y, y_pred)]
    
    return np.asarray(labels)

def pred_dset_bin_vote(fx_output, dset, fold_set):
    """
    given a likelihood classifier, predict the class of each datapoint 
    by taking the max over each observation and binary voting
    
    Parameters
    ----------
    clf : RhythmLikelihood object
        trained classifier
    dset: RhythmDataset object
        rhythm dataset 
    fold_set : str
        one of ['train','valid','test']
        
    Returns
    -------
    y_values : 2d np.ndarray
        actual and predicted classes for each datapoint in the
        fold set.
    """
    labels = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        p = fx_output(dset.reshape_batch(x))
        y_pred = np.bincount(p.argmax(axis=1)).argmax()
        labels += [(y, y_pred)]
    
    return np.asarray(labels)
    

def score(y_values):
    """
    given the output of any pred_dset_XX method, compute the
    overall accuracy
    
    Parameters
    ----------
    y_values : 2d np.ndarray
        output of the pred_dset methods
        
    Returns
    -------
    acc : float
        overall accuracy
    """
    y_act = y_values[:,0]
    y_pred = y_values[:,1]
    return (y_act==y_pred).mean()*100

def score_all(results):
    """
    compute the split scores for a set of results, where
    key="%d%s"%(one of range(k_models), one of ['train','valid','test'])
    
    Parameters
    ----------
    results : dict 
        results from any of the pred_dset_XX methods 
        
    Returns
    -------
    conf : 2d np.ndarray
        confusion matrix
    counts : 1d np.ndarray
        number of instances for each class, for normalizing the diagonal
        of the confusion matrix
    """
    Y = np.concatenate([results['%dtest'%n] for n in range(10)])
    print score(np.concatenate([results['%dtrain'%n] for n in range(10)]))
    print score(np.concatenate([results['%dtest'%n] for n in range(10)]))
    class_counts = np.asarray([(Y[:,0]==n).sum() for n in range(10)])
    return confusion_matrix(Y[:,0],Y[:,1]), class_counts
    

# - - - - - - - - Architecture - - - - - - - -  
origDimX = 60;
origDimY = 60;
arch_lmd = [AffineArgs(weight_shape=(96*120,800)),
           AffineArgs(output_shape=(400,)),
           AffineArgs(output_shape=(200,)),
           SoftmaxArgs(output_shape=(10,))]


def build_clf_model(arch=arch_lmd, scale_weights=1.0):
    """
    Convenience constructor for a RhythmLikelihood classifier. It *may* 
    help to scale the weights down for better initialization.
    
    Parameters
    ----------
    arch : list, default=arch_lmd
        architecture list
    scale_weights : float, default=1.0
        multiplier for trimming the randomly intialized weights
    
    Returns
    -------
    clf : RhythmLikelihood object
    """
    clf = RhythmLikelihood(graph=Network(arch))
    vs = clf.graph.param_values()

    for n in vs:
        for k in vs[n]:
            if k.count('W'):
                vs[n][k] *= scale_weights

    clf.graph.set_param_values(vs)

    return clf

def random_subset(dset,n_points=5000):
    """
    """
    X = []
    while len(X) < n_points:
        xi = dset.random_fetch()[0]
        X += [xi[randint(len(xi))].flatten()]
    
    return np.asarray(X)

  
# Deprecated Code ... please ignore
# ---------------------------------    
"""
arch_00 = [Conv3DArgs(input_shape=(1,689,36),weight_shape=(8,22,7),pool_shape=(4,2)),   # 689->668->167, 36->30->15
           Conv3DArgs(weight_shape=(12,20,8),pool_shape=(4,2)),                          # 167->150->37,  15->8->4 
           Conv3DArgs(weight_shape=(16,18,4),pool_shape=(2,1)),                           # 37->20->10,   4->1->1      
           AffineArgs(output_shape=(50,)),
           AffineArgs(output_shape=(3,))]

arch_01 = [Conv3DArgs(input_shape=(1,689,36),weight_shape=(8,86,7),pool_shape=(2,2)),   # 689->604->302, 36->30->15
           Conv3DArgs(weight_shape=(12,87,8),pool_shape=(2,2)),                         # 302->216->108,  15->8->4 
           Conv3DArgs(weight_shape=(16,85,4),pool_shape=(4,1)),                         # 108-> 24-> 6,   4->1->1      
           AffineArgs(output_shape=(18,)),
           AffineArgs(output_shape=(3,))]

arch_02 = [AffineArgs(weight_shape=(3600,256),update=False),
           AffineArgs(output_shape=(64,),update=False),
           AffineArgs(output_shape=(10,))]

arch_02b = [AffineArgs(weight_shape=(3600,256)),
           AffineArgs(output_shape=(64,)),
           AffineArgs(output_shape=(10,))]

arch_02c = [AffineArgs(weight_shape=(3600,256),update=False),
           AffineArgs(output_shape=(64,),update=False),
           AffineArgs(output_shape=(10,)),
           AffineArgs(output_shape=(3,))]

arch_03 = [AffineArgs(weight_shape=(3600,339)),
           AffineArgs(weight_shape=(339,32)),
           AffineArgs(output_shape=(3,))]

arch_04 = [AffineArgs(weight_shape=(3600,500)),
           AffineArgs(output_shape=(500,)),
           SoftmaxArgs(output_shape=(17,))]

arch_05 = [AffineArgs(weight_shape=(3600,256)),
           AffineArgs(output_shape=(64,)),
           SoftmaxArgs(output_shape=(17,))]

def build_pw_model(m=1.0, arch=arch_03):
    pw = Pairwise(graph=Network(arch, param_mode='forward'),
                  graph2=Network(arch, param_mode='forward'),
                  m=m)
    
    vs = pw.graph.param_values()
    for n in range(len(vs)):
        vs[n] /= 8.0
    pw.graph.set_param_values(vs)
    pw.graph2.set_param_values(vs)
    return pw

def show_coords(coords, classes):
    fig = figure()
    
    ax = fig.gca(projection='3d')
    ax.scatter3D(coords[:,0], coords[:,1], coords[:,2], c = classes)
    ax.set_xmargin(.5)
    ax.set_ymargin(.5)
    show()
    
    
colors = [[ 31, 120, 180],
         [178, 223, 138],
         [106,  61, 154],
         [  0,   0,   0],
         [225, 127,   0],
         [227,  26,  28],
         [251, 154, 153],
         [225, 225, 153],
         [177,  89,  40],
         [166, 206, 227],
         [ 51, 160,  44],
         [253, 191, 111]]
    
colors = np.asarray(colors)/256.0
    
def idx_to_colors(idxs):
    c = []
    for i in idxs:
        c += [colors[i]]
    
    return c

def project_dset(fx_proj,dset,fold_set):
    Z,Y = [],[]
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        z = fx_proj(dset.reshape_batch(x))
        Z += [z]
        Y += [y]*len(z)
        
    return np.array(Z,axis=0), np.asarray(Y)
    
def kmeans_init(arch,dset,n_points):
    clf = RhythmLikelihood(graph=Network(arch))
    
    
def knn_dset(fx_output, knn, dset, fold_set):
    labels = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        z = fx_output(dset.reshape_batch(x))
        y_pred = np.bincount(knn.predict(z)).argmax()
        labels += [(y, y_pred)]
    
    return np.asarray(labels)

def knn_dset_mean(fx_output, knn, dset, fold_set):
    labels = []
    for i,y in dset.split_idx[fold_set]:
        x,yi = dset.get(i)
        assert yi==y
        z = fx_output(dset.reshape_batch(x))
        y_pred = np.bincount(knn.predict(z.mean(axis=0)[np.newaxis,:])).argmax()
        labels += [(y, y_pred)]
    
    return np.asarray(labels)


def sweep_knn(z,y,fx_output,dset,k_neighbors=[1],fold_set='train'):
    for k in k_neighbors:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
        knn.fit(z,y)
        y_pairs = knn_dset(fx_output, knn, dset, fold_set)
        s = score(y_pairs)
        print "K: %d \t Acc:%0.3f"%(k,s)
"""
