#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GMM
from subprocess import Popen, PIPE
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

genreNames = [u'Axé', 'Bachata', 'Bolero',
          u'Forró', u'Gaúcha', 'Merengue',
            'Pagode', 'Salsa', 'Sertaneja', 'Tango']

cmap= plt.get_cmap('spectral', 10)
for i in range(10):
    colors[i]=cmap(i)

def tempoGaus(tempoList, gPlot=range(10), acc=2, hist=False, lw=4):
    #takes a list of tempos divided into tempo arrays for each 01 10 genres

    fig = plt.figure(figsize=(10, 3.3))
    fig.subplots_adjust(left=0.1, right=0.97,
                                bottom=0.17, top=0.9, wspace=0.35)


    # plot 1: data + best-fit mixture
    ax = fig.add_subplot(111)


        
    for i in gPlot:
        X=np.array(tempoList[i])

        #------------------------------------------------------------
        # Learn the best-fit GMM models
        #  Here we'll use GMM in the standard way: the fit() method
        #  uses an Expectation-Maximization approach to find the best
        #  mixture of Gaussians for the data

        # fit models with 1-10 components
        np.random.seed(1)
        N = np.arange(1, acc)
        models = [None for x in range(len(N))]

        for j in range(len(N)):

            models[j] = GMM(N[j]).fit(X)

        # compute the AIC and the BIC
        AIC = [m.aic(X) for m in models]
        BIC = [m.bic(X) for m in models]

        #------------------------------------------------------------
        # Plot the results
        #  We'll use three panels:
        #   1) data + best-fit mixture
        #   2) AIC and BIC vs number of components
        #   3) probability that a point came from each component
        M_best = models[np.argmin(AIC)]

        x = np.linspace(0, 400, len(X))
        logprob, responsibilities = M_best.eval(x)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
       
        if hist: 
            ax.hist(X, 30, normed=True, histtype='stepfilled', alpha=0.2)
        ax.plot(x, pdf, color=colors[i],label=genreNames[i], linewidth=lw, alpha=0.7)
        #ax.plot(x, pdf_individual, '--k')

    ax.set_xlabel('BPM')
    ax.set_ylabel('%')


    handles, labels = ax.get_legend_handles_labels()
    #shrink text
    fontP = FontProperties()
    fontP.set_size('small')
    #shrink box to fit legend on left
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels)#,
    '''
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
            prop = fontP);
    '''
    rect = fig.patch
    rect.set_facecolor('white')

    plt.show()

def play(filename):

    try:
        p = Popen(["play", filename, 'trim', '60'],
        stdout=PIPE)

    except KeyboardInterrupt:
        raise
    except:
        print "exiting" 
        p.send_signal(signal.SIGINT)
        p.wait()
        return
    return
        # report error and proceed
