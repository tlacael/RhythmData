import lmd as L
from score_save import gather_scores
import os
import time as T

#T.sleep(6*60*60)
#set directories
topDir = "/Users/Tlacael/NYU/RhythmData/"
dsetP1 = "lmd240x10_6oct.hdf5"
scal1 = "lmd_scalars240x10.pkl"

expName = 'neuronTest_c-240x10/'
T.sleep(60*60)
#### test 128x60 arch
dims = [240,10]
startSize = dims[0]*dims[1]
lay1 = 1024
lay2 = 32


arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*4,)),
        L.AffineArgs(output_shape=(lay2*2,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp12-n-1-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
##########################################

arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp12-n-2-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*8,)),
        L.AffineArgs(output_shape=(lay2*4,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp12-n-3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*4,)),
        L.AffineArgs(output_shape=(lay2*4,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp12-n-3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
#
