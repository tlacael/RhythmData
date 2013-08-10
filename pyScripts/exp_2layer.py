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

#### test 128x60 arch
dims = [240,10]
startSize = dims[0]*dims[1]
lay1 = 1024
lay2 = 32
'''
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp11-n-1-fold'
for i in range(10):
        curPreFix = fPreFix + str(i)
        gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
                        
##########################################

startSize = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*2,)),
        L.SoftmaxArgs(output_shape=(10,))]
  
   
fPreFix = 'exp11-n0-fold'
for i in range(10):
        curPreFix = fPreFix + str(i)
        gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix) 
                    
##########################################

startSize = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*4,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp11-n1-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*8,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp11-n2-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*16,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp11-n3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
'''
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2*32,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp11-n4-fold'
for i in range(0,4):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################

arch = arch = [L.AffineArgs(weight_shape=(startSize,lay1)),
        L.AffineArgs(output_shape=(lay2/2,)),
        L.SoftmaxArgs(output_shape=(10,))]

'''
fPreFix = 'exp11-n5-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)


##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize*2)),
        L.AffineArgs(output_shape=(startSize/16,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp4-n7-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
'''
