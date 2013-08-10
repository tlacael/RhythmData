import lmd as L
from score_save import gather_scores
import os

#set directories
topDir = "/Users/Tlacael/NYU/RhythmData/"
dsetP1 = "lmd32x60_6oct.hdf5"
scal1 = "lmd_scalars32x60.pkl"

expName = 'NeuronTest_a/'

#### test 128x60 arch
dims = [32,60]
startSize = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize/4)),
        L.AffineArgs(output_shape=(startSize/8,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n1-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize/2)),
        L.AffineArgs(output_shape=(startSize/4,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n2-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize/2)),
        L.AffineArgs(output_shape=(startSize/8,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n3-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize/8)),
        L.AffineArgs(output_shape=(startSize/16,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n4-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize/8)),
        L.AffineArgs(output_shape=(startSize/32,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n5-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize*2)),
        L.AffineArgs(output_shape=(startSize/8,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n6-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)
##########################################
arch = arch = [L.AffineArgs(weight_shape=(startSize,startSize*2)),
        L.AffineArgs(output_shape=(startSize/16,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp3-n7-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)

