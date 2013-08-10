import lmd2 as L
import time as T
import lmd as L2
import alt_gatherScalars as G

t=0
while(t<170):
    T.sleep(60)
    t+=1
    print str(45-t)+ ' minutes left'
print 'get scalars'

G.getScalars('/Users/Tlacael/Documents/MATLAB/LMDdata/LMD960x10_6oct/', [960,10])

print 'import data'
dim1=960
dim2=10
hdf = '/Users/Tlacael/NYU/RhythmData/lmd'+str(dim1)+'x'+str(dim2)+'_6oct.hdf5'
scalar = '/Users/Tlacael/NYU/RhythmData/lmd_scalars'+str(dim1)+'x'+str(dim2)+'.pkl'
rootDir = '/Users/Tlacael/Documents/MATLAB/LMDdata/'
OPdir = '/Users/Tlacael/Documents/MATLAB/LMDdata/LMD420x10_6oct/'

#G.getScalars('/Users/Tlacael/Documents/MATLAB/LMDdata/LMD420x10_6oct/', [dim1,dim2])




L.import_data('/Users/Tlacael/NYU/RhythmData/lmd'+str(dim1)+'x'+str(dim2)+'_6oct.hdf5',0,'/Users/Tlacael/Documents/MATLAB/LMDdata/', dname='LMD'+str(dim1)+'x'+str(dim2)+'_6oct')

dset = L2.RhythmDataset('/Users/Tlacael/NYU/RhythmData/lmd_scalars'+str(dim1)+'x'+str(dim2)+'.pkl','/Users/Tlacael/NYU/RhythmData/lmd'+str(dim1)+'x'+str(dim2)+'_6oct.hdf5',valid=6,test=3, dim=[dim1,dim2])


dset.stratify(10)


