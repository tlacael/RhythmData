import lmd as L
from score_save import gather_scores
import os
import time as T

print 'sleeping'
#T.sleep(6*60*60)
#set directories
topDir = "/Users/Tlacael/NYU/RhythmData/"
dsetP1 = "lmd240x10_6oct.hdf5"
scal1 = "lmd_scalars240x10.pkl"

expName = 'neuronTest_c-240x10/'

#### test 128x60 arch
dims = [240,10]
startSize = dims[0]*dims[1]
'''
arch = arch = [L.AffineArgs(weight_shape=(startSize,64)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp8-n-1-fold'
for i in range(10):
        curPreFix = fPreFix + str(i)
        gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
                        
##########################################

startSize = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(startSize,128)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp8-n0-fold'
for i in range(10):
        curPreFix = fPreFix + str(i)
        gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
                    
##########################################

startSize = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(startSize,256)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp8-n1-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,256*2)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp8-n2-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,256*4)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp8-n3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
'''
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,256*8)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp8-n4-fold'
for i in range(3,10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################

arch = arch = [L.AffineArgs(weight_shape=(startSize,256*16)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp8-n5-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

'''
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize*2)),
        L.AffineArgs(output_shape=(startSize/16,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp4-n7-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
'''
