import csv
import numpy as np
from matplotlib import pyplot as plt


class MLPlayerGraph():
    def __init__(self, filename):
        self.data = loadCSVdata(filename)
        for m in self.data:
            if m[5]=='-':
                m[5]='0 layers - linear'
                print '?'+m[5]+'?'
    
    def scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(range(len(self.data)), [float(m[4]) for m in self.data], label=[m[5] for m in self.data])
        ax.set_xticks(range(len(self.data)))
        ax.set_xticklabels([m[5] for m in self.data], rotation='vertical', size = 'smaller')
        #ax.set_title("Results of MLP Layer Depth and Neuron Size Experiments (240x10 OP)")
        ax.set_ylabel('Mean Accuracy (%)')
        ax.set_xlabel('Architecture')
        #ax.viewLim([[ 0,0 ],[len(m) ,100 ]])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0+0.1, box.width, box.height*0.8])


        ax.errorbar(range(len(self.data)), [float(m[4]) for m in self.data], [float(m[6])/2. for m in self.data], linestyle="None", marker="None", color="green")

        rect = fig.patch
        rect.set_facecolor('white')

        
        plt.show()


    

def loadCSVdata(filename):
    fid = open(filename, 'rU')
    r = csv.reader(fid, delimiter=',')
    
    temp=[]
    mat = []
    for row in r:
        temp = np.array(row).reshape((1,len(row)))

        if not isinstance(mat, np.ndarray):
            mat = temp
        else:
            mat = np.concatenate((mat,temp),axis=0)
    return mat
