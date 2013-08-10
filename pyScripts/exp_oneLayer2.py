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

##########################################

arch = [L.AffineArgs(weight_shape=(startSize,256*5)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n1-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*6)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n2-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*7)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*8)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n4-fold'
for i in range(0,3):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################

arch = [L.AffineArgs(weight_shape=(startSize,256*9)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n5-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)


##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*10)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n6-fold'
for i in range(2,10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*11)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n7-fold'
for i in range(2,10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = [L.AffineArgs(weight_shape=(startSize,256*12)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp14-n8-fold'
for i in range(2,10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

