import lmd as L
from score_save import gather_scores
import os
import time as T

#set directories
topDir = "/Users/Tlacael/NYU/RhythmData/"
dsetP1 = "lmd60x120_6oct.hdf5"
dsetP2 = "hdfFiles/30x10_mid.hdf5"
dsetP3 = "hdfFiles/LMD240x25_np_zp.hdf5"
scal1 = "lmd_scalars60x120.pkl"
scal2 = "scalarFiles/LMD_scalars30x10.pkl"
scal3 = "scalarFiles/LMD_scalars240x25.pkl"

expName = 'CQTest/'

#### test 128x60 arch
dims = [30,10]
arch = arch = [
        L.AffineArgs(weight_shape=(dims[0]*dims[1],256)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]

fPreFix = 'exp2try-30x10-mid'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP2,arch,topDir+scal2,dims, i, curPreFix, True,'clfFiles/'+fPreFix+'_'+str(i)+'.clf')

#### test 64x60 arch
'''
dims = [64,60]
arch = arch = [L.AffineArgs(weight_shape=(dims[0]*dims[1],256)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]

fPreFix = 'exp1-64x60_6oct--fold'
for i in range(8,10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP2,arch,topDir+scal2,dims, i, curPreFix)

#### test 32x60 arch
dims = [32,60]
arch = arch = [L.AffineArgs(weight_shape=(dims[0]*dims[1],256)),
        L.AffineArgs(output_shape=(64,)),
        L.SoftmaxArgs(output_shape=(10,))]

fPreFix = 'exp1-32x60-fold'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP3,arch,topDir+scal3,dims, i, curPreFix)

#### test 128x60 arch
dims = [128,60]
d1 = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(d1,d1/16)),
        L.AffineArgs(output_shape=(d1/32,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp2-128x60-fold_6oct'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP1,arch,topDir+scal1,dims, i, curPreFix)


#### test 64x60 arch
dims = [64,60]
d1 = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(d1,d1/16)),
        L.AffineArgs(output_shape=(d1/32,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp2-64x60-fold_6oct'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP2,arch,topDir+scal2,dims, i, curPreFix)


#### test 64x60 arch
dims = [32,60]
d1 = dims[0]*dims[1]
arch = arch = [L.AffineArgs(weight_shape=(d1,d1/16)),
        L.AffineArgs(output_shape=(d1/32,)),
        L.SoftmaxArgs(output_shape=(10,))]


fPreFix = 'exp2-32x60-fold_6oct'
for i in range(10):
    curPreFix = fPreFix + str(i)
    gather_scores(topDir+dsetP3,arch,topDir+scal3,dims, i, curPreFix)
'''
