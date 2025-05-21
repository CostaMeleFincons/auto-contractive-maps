# -*- coding: utf-8 -*-
"""
This file is part of ac-maps.
ac-maps is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
ac-maps is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with ac-maps.  If not, see <http://www.gnu.org/licenses/>.
"""



from datetime import datetime
import os
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import jinja2
import csv

import sys
sys.path.insert(0, '../helper')
import mrg
import dataset



class Acm:
    ''' This class implements an Auto Contractive Map.
    '''

    def __init__(self, _inputLength, _contraction, _dataset, _datasetPath='', _scale='column'):
        ''' Initialization function.

            Arguments:
                _inputLength (int): Length of input vector
                _contraction (float): Contraction parameter, _contraction>1.
                _dataset (str): Name/path/folder of dataset to use: 'random', 'correlated1', 'correlated2', or 'mnist'.
        '''
        # Length of input vector
        self.N = _inputLength

        # Contraction parameter
        self.C = _contraction

        # Labels of training data
        self.label = ['' for x in range(self.N)]

        # Throw exception, if computation fails
        np.seterr('raise')

        # Runtime
        self.cntFinal = []

        # Weights
        self.wFinal = []

        # Weights
        self.vFinal = []

        # MST
        self.mstFinal = []

        # MRG
        self.mrgFinal = []

        # PCA
        self.pcaFinal = []

        # PCA
        self.pearsonrFinal = []

        # MRG class object
        self.cMrg = mrg.Mrg()

        # Dataset class object
        self.cDataset = dataset.Dataset()

        # Dataset to use
        self.dataset = _dataset

        # Pathto Dataset
        self.datasetPath = _datasetPath

        # Scale to use
        self.scale = _scale

        # Reset
        self.resetNN()



    def resetNN(self):
        ''' Resets variables for NN.
        '''
        # First layer
        self.v = np.full((1, self.N), 0.001, dtype=float)[0]
 
        # Hidden layer
        self.w = np.full((self.N, self.N), 0.001, dtype=float)
        
        # Neuron values
        self.mHidden = np.zeros((1, self.N), dtype=float)[0]
        self.mOut = np.zeros((1, self.N), dtype=float)[0]

        # Training data
        self.training = np.zeros((1, self.N), dtype=float)

        # Result matrices
        self.wMean = np.zeros((self.N, self.N), dtype=float)
        self.wStd = np.zeros((self.N, self.N), dtype=float)
        self.vMean = np.zeros((1, self.N), dtype=float)[0]
        self.vStd = np.zeros((1, self.N), dtype=float)[0]
        self.pcaExplainedVarMean = np.zeros((1, self.N), dtype=float)[0]
        self.pcaExplainedVarStd = np.zeros((1, self.N), dtype=float)[0]
        self.pearsonrMean = np.zeros((self.N, self.N, 2), dtype=float)
        self.pearsonrStd = np.zeros((self.N, self.N, 2), dtype=float)



    def runOnce(self, _mIn):
        ''' This function performs one run of training using _mIn as input vector.

            Arguments:
                _mIn (np.array(dtype=float)): Input vector.
        '''
        mIn = _mIn

        # 1. Signal In to Hidden
        for i in range(self.N):
            self.mHidden[i] = mIn[i] * (1 - self.v[i]/self.C)

        # 2. Adapt weights In to Hidden (v)
        # The formula is actualle m_s * (1 - (v/C)^2)
        for i in range(self.N):
            self.v[i] += (mIn[i] - self.mHidden[i]) * (1 - (self.v[i]/self.C))

        # 3. Signal Hidden to Out
        self.net = np.zeros((1, self.N), dtype=float)[0]
        for i in range(self.N):
            for j in range(self.N):
                self.net[i] += self.mHidden[j] * (1 - (self.w[i][j]/self.C))

        for i in range(self.N):
            self.mOut[i] = self.mHidden[i] * (1 - self.net[i]/self.C)

        # 4. Adapt weights Hidden to Out (w)
        for i in range(self.N):
            for j in range(self.N):
                # Perform computation in single steps to avoid overflow
                self.w[i][j] += (self.mHidden[i] - self.mOut[i]) * (1 - self.w[i][j]/self.C) * self.mHidden[j]



    def scaleTraining(self):
        ''' This function scales the training data according to self.scale.

            self.scale may have one of the following values: 'rows', 'columns', or 'table'.
        '''
        if self.scale == 'rows':
            # Scale row wise
            for cnt in range(len(self.training)):
                row = self.training[cnt]
                self.training[cnt] = np.interp(row, (row.min(), row.max()), (0, 1))
        elif self.scale == 'columns':
            # Scale column wise
            self.training = self.training.transpose()
            for cnt in range(len(self.training)):
                row = self.training[cnt]
                self.training[cnt] = np.interp(row, (row.min(), row.max()), (0, 1))
            self.training = self.training.transpose()
        elif self.scale == 'table':
            # Scale table wise
            tableMin = np.min(self.training)
            tableMax = np.max(self.training)

            for cnt in range(len(self.training)):
                row = self.training[cnt]
                self.training[cnt] = np.interp(row, (tableMin, tableMax), (0, 1))



    def run(self, _nr=1):
        ''' This function trains the map given the training samples in self.training.

            Arguments:
                _nr (int): Number of runs _nr >= 1.
        '''
        # Perform _nr runs
        cntNr = 0
        while cntNr < _nr:
            print('Running: ' + str(cntNr) + '/' + str(_nr))

            # Reset Neural Net
            self.resetNN()

            # Create training samples
            if os.path.isdir(self.datasetPath):
                self.training, self.label = self.cDataset.loadDataset(self.datasetPath, cntNr)
            elif os.path.isfile(self.datasetPath):
                self.training, self.label = self.cDataset.loadFile(self.datasetPath)
            else:
                if self.dataset.lower() == 'random':
                    self.training, self.label = self.cDataset.createTrainingRandom(self.N)
                elif self.dataset.lower() == 'correlated1':
                    self.training, self.label = self.cDataset.createTrainingCorrelated1(self.N)
                elif self.dataset.lower() == 'correlated2':
                    self.training, self.label = self.cDataset.createTrainingCorrelated2(self.N)
                else:
                    raise ValueError('Dataset not found.', self.datasetPath, self.dataset)

            # Scale data
            self.scaleTraining()

            # Runtime counter
            cnt = 0
            successfull = True
            for x in self.training:
                # For random data, the self.mOut will sometimes run away and will create a buffer overflow
                '''
                if any(np.greater(self.mOut, 1e+10)) is True or \
                        any(np.less(self.mOut, -1e+10)) is True:
                    successfull = False
                    print('Float over or underflow in ' + str(cntNr))
                    cntNr -= 1
                    break
                '''

                #if len(self.cntFinal) > 1:
                #    print('a', self.cntFinal[-1], self.mOut)
                self.runOnce(x)
                cnt += 1
 
                # Check, if training is finished
                # After that output oscillates and may throw RuntimeWarning:Overflow encounter
                maxOut = np.amax(self.mOut)
                minOut = np.amin(self.mOut)
                if minOut >= 0  and maxOut < 1e-6:
                    break
                if self.dataset == 'mnist':
                    print(cntNr, cnt, np.mean(self.mOut), minOut, maxOut)

            # Append results
            if successfull:
                self.mstFinal.append(minimum_spanning_tree(self.w))
                self.mrgFinal.append(self.cMrg.computeMrg(self.w)[0])
                self.cntFinal.append(cnt)
                self.wFinal.append(self.w)
                self.vFinal.append(self.v)
                self.pcaFinal.append(self.runPca())
                # Compute Pearson's r
                pearsonrData = np.array(self.training[0:cnt]).T
                pearsonrOut = np.zeros((self.N, self.N, 2), dtype=float)
                for i in range(self.N):
                    for j in range(self.N):
                        pearsonrOut[i][j] = pearsonr(pearsonrData[i], pearsonrData[j])
                self.pearsonrFinal.append(pearsonrOut)

            cntNr += 1

        # Compute results
        # weights w
        # Numpy seems to have trouble with a list of 2d matrices.
        for i in range(self.N):
            for j in range(self.N):
                row = []
                for k in range(len(self.cntFinal)):
                    row.append(self.wFinal[k][i][j])
                self.wMean[i][j] = np.mean(row)
                self.wStd[i][j] = np.std(row)
        # weights v
        for i in range(self.N):
            row = []
            for k in range(len(self.cntFinal)):
                row.append(self.vFinal[k][i])
            self.vMean[i] = np.mean(row)
            self.vStd[i] = np.std(row)

        # PCA explained variance
        for i in range(self.N):
            row = []
            for k in range(len(self.cntFinal)):
                row.append(self.pcaFinal[k].explained_variance_ratio_[i])
            self.pcaExplainedVarMean[i] = np.mean(row)
            self.pcaExplainedVarStd[i] = np.std(row)

        # Pearson's r
        for i in range(self.N):
            for j in range(self.N):
                row0 = []
                row1 = []
                for k in range(len(self.cntFinal)):
                    # Replace -1e-200 < values < 1e-200 with 0. Otherwise numpy throws errors
                    if self.pearsonrFinal[k][i][j][0] > float(-1e-200) and self.pearsonrFinal[k][i][j][0] < float(1e-200):
                        row0.append(0)
                    else:
                        row0.append(self.pearsonrFinal[k][i][j][0])
                    if self.pearsonrFinal[k][i][j][1] > float(-1e-200) and self.pearsonrFinal[k][i][j][1] < float(1e-200):
                        row1.append(0)
                    else:
                        row1.append(self.pearsonrFinal[k][i][j][1])
                self.pearsonrMean[i][j] = [np.mean(row0), np.mean(row1)]
                self.pearsonrStd[i][j] = [np.std(row0), np.std(row1)]



    def runPca(self):
        ''' Computes a PCA on the training data.

            Returns:
                (sklearn.decomposition.PCA). PCA
        '''
        trainXStd = StandardScaler().fit_transform(self.training)
        pca = PCA(n_components=self.N, svd_solver='full')
        pca.fit(trainXStd)
        return pca



    def printTree(self):
        ''' This function prints the last results of self.run().
        '''

        print('MST:')
        corr = self.mstFinal[-1].toarray().astype(float)
        for i in range(self.N):
            for j in range(self.N):
                if not corr[i][j] == 0:
                    print('Connection: {0}\t--> {1}\tWeight: {2:.3f}'.format(
                        self.label[i],
                        self.label[j],
                        corr[i][j]))
        print('MRG:')
        corr = self.mrgFinal[-1]
        for i in range(self.N):
            for j in range(self.N):
                if not corr[i][j] == 0:
                    print('Connection: {0}\t--> {1}\tWeight: {2:.3f}'.format(
                        self.label[i],
                        self.label[j],
                        corr[i][j]))
        print('')



    def printStatistics(self):
        ''' This function prints statistics over all runs.
        '''
        print('Total number of runs: ' + str(len(self.cntFinal)))
        print('Mean training length: {0:.2f} +- {1:.2f} ({2:.2f}%)'.format(
                np.mean(self.cntFinal),
                np.std(self.cntFinal),
                100/np.mean(self.cntFinal)*np.std(self.cntFinal)))

        print('Highest H score of Mean weights w: ' + str(self.cMrg.computeMrg(self.wMean)[1]))

        #print('Mean weights w: ' + str(self.wMean))
        #print('Std weights w: ' + str(self.wStd))

        print('Mean weights w: ' + str(np.mean(self.wMean)))
        print('Std weights w: ' + str(np.std(self.wStd)))



    def draw(self):
        ''' This function draws the tree, which results from the last run of self.run().
        '''
        # MST
        self.createGraph(self.mstFinal[-1].toarray().astype(float))
        plt.show()
        plt.clf()

        # MRG
        self.createGraph(self.mrgFinal[-1])
        plt.show()
        plt.clf()



    def createGraph(self, _w):
        ''' Creates a networkx graph out of a matrix in a matplotlib figure.

            Arguments:
                _w (np.array): Input matrix.
        '''
        cellFrom = []
        cellTo = []
        corrEdge = []
        G = nx.Graph() 
        for i in range(self.N):
            for j in range(self.N):
                if not _w[i][j] == 0:
                    cellFrom.append(self.label[i])
                    cellTo.append(self.label[j])
                    corrEdge.append(_w[i][j])
                    G.add_edge(self.label[i], self.label[j], weight=_w[i][j]*10000)
        nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=10, font_size=15)



    def save(self, _folderOut, _pathTemplate):
        ''' This function saves all results to file.

            Arguments:
                _folderOut (str): All output files will be placed in this folder.
                _pathTemplate (str): Path to jinja2 template for gnuplot weight heat map.
        '''
        # Precision for data when saved
        precision = 4

        # Filenames
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = str(date) + '_dataset-' + str(self.dataset) + '_nrruns-' + str(len(self.cntFinal))
        filenameWeights = os.path.join(_folderOut, filename + '_weights.txt')
        filenameWeightsPlot = os.path.join(_folderOut, filename + '_weights.plot')
        filenameWeightsPng = os.path.join(filename + '_weights.png')
        filenameWeightsTex = os.path.join(filename + '_weights.tex')
        filenameWeightsMean = os.path.join(_folderOut, filename + '_weightsmean.txt')
        filenameWeightsMeanPlot = os.path.join(_folderOut, filename + '_weightsmean.plot')
        filenameWeightsMeanPng = os.path.join(filename + '_weightsmean.png')
        filenameWeightsMeanTex = os.path.join(filename + '_weightsmean.tex')
        filenameGraphMst = os.path.join(_folderOut, filename + '_mst.png')
        filenameGraphMstMean = os.path.join(_folderOut, filename + '_mstmean.png')
        filenameGraphMrg = os.path.join(_folderOut, filename + '_mrg.png')
        filenameGraphMrgMean = os.path.join(_folderOut, filename + '_mrgmean.png')
        filenamePca = os.path.join(_folderOut, filename + '_pca.txt')
        filenamePcaPlot = os.path.join(_folderOut, filename + '_pca.plot')
        filenamePcaPng = os.path.join(filename + '_pca.png')
        filenamePcaTex = os.path.join(filename + '_pca.tex')
        filenamePcaMean = os.path.join(_folderOut, filename + '_pcamean.txt')
        filenamePcaMeanPlot = os.path.join(_folderOut, filename + '_pcamean.plot')
        filenamePcaMeanPng = os.path.join(filename + '_pcamean.png')
        filenamePcaMeanTex = os.path.join(filename + '_pcamean.tex')
        filenamePearsonr = os.path.join(_folderOut, filename + '_pearsonr.txt')
        filenamePearsonrPlot = os.path.join(_folderOut, filename + '_pearsonr.plot')
        filenamePearsonrPng = os.path.join(filename + '_pearsonr.png')
        filenamePearsonrTex = os.path.join(filename + '_pearsonr.tex')
        filenamePearsonrMean = os.path.join(_folderOut, filename + '_pearsonrmean.txt')
        filenamePearsonrMeanPlot = os.path.join(_folderOut, filename + '_pearsonrmean.plot')
        filenamePearsonrMeanPng = os.path.join(filename + '_pearsonrmean.png')
        filenamePearsonrMeanTex = os.path.join(filename + '_pearsonrmean.tex')
        filenamePearsonr2 = os.path.join(_folderOut, filename + '_pearsonr2.txt')
        filenamePearsonr2Plot = os.path.join(_folderOut, filename + '_pearsonr2.plot')
        filenamePearsonr2Png = os.path.join(filename + '_pearsonr2.png')
        filenamePearsonr2Tex = os.path.join(filename + '_pearsonr2.tex')
        filenamePearsonr2Mean = os.path.join(_folderOut, filename + '_pearsonr2mean.txt')
        filenamePearsonr2MeanPlot = os.path.join(_folderOut, filename + '_pearsonr2mean.plot')
        filenamePearsonr2MeanPng = os.path.join(filename + '_pearsonr2mean.png')
        filenamePearsonr2MeanTex = os.path.join(filename + '_pearsonr2mean.tex')
        filenamePickle = os.path.join(_folderOut, filename + '_net.p')

        # Save weights of last run
        self.writeFile(filenameWeights, np.around(self.w, decimals=precision), _header=['#i', 'j', 'Weight'])

        # Create gnuplot scripts from jinja2 template of last run
        templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenameWeightsPng, 
                filenameWeightsTex=filenameWeightsTex,
                filenameWeights=os.path.basename(filenameWeights),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange=str(int(np.amin(self.w))) + ':' + str(int(np.amax(self.w))+1),
                using='1:2:(sprintf("%.' + str(precision) + 'f",$3))')

        fp = open(filenameWeightsPlot, 'w')
        fp.write(script)
        fp.close()

        # Save weights mean
        self.writeFile(filenameWeightsMean, np.around(self.wMean, decimals=precision), np.around(self.wStd, decimals=precision), _header=['#i', 'j', 'Mean', 'Std'])

        # Create gnuplot scripts from jinja2 template mean
        templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenameWeightsMeanPng, 
                filenameWeightsTex=filenameWeightsMeanTex,
                filenameWeights=os.path.basename(filenameWeightsMean),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange=str(int(np.amin(self.wMean)*10)/10) + ':' + str(int(np.amax(self.wMean)*10)/10+0.1),
                using='1:2:(sprintf("\\\\tiny{$%.' + str(precision) + 'f \\\\pm %.' + str(precision) + 'f$}", $3, $4))')

        fp = open(filenameWeightsMeanPlot, 'w')
        fp.write(script)
        fp.close()

        # Save graph of last run MST
        self.createGraph(self.mstFinal[-1].toarray().astype(float))
        plt.savefig(filenameGraphMst, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait')
        plt.clf()

        # Save graph of last run MRG
        self.createGraph(self.mrgFinal[-1])
        plt.savefig(filenameGraphMrg, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait')
        plt.clf()

        # Save graph mean MST
        self.createGraph(minimum_spanning_tree(self.wMean).toarray().astype(float))
        plt.savefig(filenameGraphMstMean, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait')
        plt.clf()

        # Save graph mean MRG
        self.createGraph(self.cMrg.computeMrg(self.wMean)[0])
        plt.savefig(filenameGraphMrgMean, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait')
        plt.clf()

        # PCA of last run
        if len(self.pcaFinal) > 0:
            self.writeFile(filenamePca, np.around(self.pcaFinal[-1].explained_variance_ratio_, decimals=precision), _header=['#i', 'j', 'ExplainedVariance'])

            # Create gnuplot scripts from jinja2 template of last run
            templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
            templateEnv = jinja2.Environment(loader=templateLoader)
            template = templateEnv.get_template('pca.plot.jinja2')
            script = template.render(filenamePcaPng=filenamePcaPng, 
                    filenamePcaTex=filenamePcaTex,
                    filenamePca=os.path.basename(filenamePca),
                    range=str(-0.5) + ':' + str(self.N-1+0.5),
                    using='3',
                    errorbars='')

            fp = open(filenamePcaPlot, 'w')
            fp.write(script)
            fp.close()

        # PCA mean
        self.writeFile(filenamePcaMean, np.around(self.pcaExplainedVarMean, decimals=precision), _data2=np.around(self.pcaExplainedVarStd, decimals=precision), _header=['#i', 'j', 'Mean', 'Std'])

        # Create gnuplot scripts from jinja2 template mean
        templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('pca.plot.jinja2')
        script = template.render(filenamePcaPng=filenamePcaMeanPng, 
                filenamePcaTex=filenamePcaMeanTex,
                filenamePca=os.path.basename(filenamePcaMean),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                using='3:4',
                errorbars='errorbars gap 2 lw 3')

        fp = open(filenamePcaMeanPlot, 'w')
        fp.write(script)
        fp.close()

        # Pearson's r of last run
        if len(self.pearsonrFinal) > 0:
            pearsonrMean = np.zeros((self.N, self.N), dtype=float)
            for i in range(self.N):
                for j in range(self.N):
                    pearsonrMean[i][j] = self.pearsonrFinal[-1][i][j][0]
            self.writeFile(filenamePearsonr, np.around(pearsonrMean, decimals=precision), _header=['#i', 'j', 'r1'])
 
            # Create gnuplot scripts from jinja2 template of last run
            templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
            templateEnv = jinja2.Environment(loader=templateLoader)
            template = templateEnv.get_template('weights.plot.jinja2')
            script = template.render(filenameWeightsPng=filenamePearsonrPng, 
                    filenameWeightsTex=filenamePearsonrTex,
                    filenameWeights=os.path.basename(filenamePearsonr),
                    range=str(-0.5) + ':' + str(self.N-1+0.5),
                    cbrange='-1:1',
                    using='1:2:(sprintf("%.' + str(precision) + 'f",$3))')
 
            fp = open(filenamePearsonrPlot, 'w')
            fp.write(script)
            fp.close()

        # Pearson's r mean
        pearsonrMean = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                pearsonrMean[i][j] = self.pearsonrMean[i][j][0]
        pearsonrStd = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                pearsonrStd[i][j] = self.pearsonrStd[i][j][0]
        self.writeFile(filenamePearsonrMean, np.around(pearsonrMean, decimals=precision), np.around(pearsonrStd, decimals=precision), _header=['#i', 'j', 'Mean', 'Std'])

        # Create gnuplot scripts from jinja2 template mean
        templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenamePearsonrMeanPng, 
                filenameWeightsTex=filenamePearsonrMeanTex,
                filenameWeights=os.path.basename(filenamePearsonrMean),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange='-1:1',
                using='1:2:(sprintf("\\\\tiny{$%.' + str(precision) + 'f \\\\pm %.' + str(precision) + 'f$}", $3, $4))')

        fp = open(filenamePearsonrMeanPlot, 'w')
        fp.write(script)
        fp.close()

        # Pearson's r of last run 2
        if len(self.pearsonrFinal) > 0:
            pearsonrMean = np.zeros((self.N, self.N), dtype=float)
            for i in range(self.N):
                for j in range(self.N):
                    pearsonrMean[i][j] = self.pearsonrFinal[-1][i][j][1]
            self.writeFile(filenamePearsonr2, np.around(pearsonrMean, decimals=precision), _header=['#i', 'j', 'r2'])
 
            # Create gnuplot scripts from jinja2 template of last run
            templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
            templateEnv = jinja2.Environment(loader=templateLoader)
            template = templateEnv.get_template('weights.plot.jinja2')
            script = template.render(filenameWeightsPng=filenamePearsonr2Png, 
                    filenameWeightsTex=filenamePearsonr2Tex,
                    filenameWeights=os.path.basename(filenamePearsonr2),
                    range=str(-0.5) + ':' + str(self.N-1+0.5),
                    cbrange='-1:1',
                    using='1:2:(sprintf("%.' + str(precision) + 'f",$3))')
 
            fp = open(filenamePearsonr2Plot, 'w')
            fp.write(script)
            fp.close()

        # Pearson's r mean 2
        pearsonrMean = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                pearsonrMean[i][j] = self.pearsonrMean[i][j][1]
        pearsonrStd = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            for j in range(self.N):
                pearsonrStd[i][j] = self.pearsonrStd[i][j][1]
        self.writeFile(filenamePearsonr2Mean, np.around(pearsonrMean, decimals=precision), np.around(pearsonrStd, decimals=precision), _header=['#i', 'j', 'Mean', 'Std'])

        # Create gnuplot scripts from jinja2 template mean
        templateLoader = jinja2.FileSystemLoader(searchpath=_pathTemplate)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenamePearsonr2MeanPng, 
                filenameWeightsTex=filenamePearsonr2MeanTex,
                filenameWeights=os.path.basename(filenamePearsonr2Mean),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange='-1:1',
                using='1:2:(sprintf("\\\\tiny{$%.' + str(precision) + 'f \\\\pm %.' + str(precision) + 'f$}", $3, $4))')

        fp = open(filenamePearsonr2MeanPlot, 'w')
        fp.write(script)
        fp.close()
        # Save all weights
        saveNet = {}
        saveNet['N'] = self.N
        saveNet['C'] = self.C
        saveNet['label'] = self.label
        saveNet['dataset'] = self.dataset
        saveNet['datasetPath'] = self.datasetPath
        saveNet['cntFinal'] = self.cntFinal
        saveNet['wFinal'] = self.wFinal
        saveNet['vFinal'] = self.vFinal
        saveNet['mstFinal'] = self.mstFinal
        saveNet['mrgFinal'] = self.mrgFinal
        saveNet['pcaFinal'] = self.pcaFinal
        saveNet['wMean'] = self.wMean
        saveNet['wStd'] = self.wStd
        saveNet['vMean'] = self.vMean
        saveNet['vStd'] = self.vStd
        saveNet['training'] = self.training
        saveNet['pcaExplainedVarMean'] = self.pcaExplainedVarMean
        saveNet['pcaExplainedVarStd'] = self.pcaExplainedVarStd
        saveNet['pearsonrMean'] = self.pearsonrMean
        saveNet['pearsonrStd'] = self.pearsonrStd
        saveNet['pearsonrFinal'] = self.pearsonrFinal
        pickle.dump(saveNet, open(filenamePickle, "wb" ))



    def load(self, _pathPickle):
        ''' This function loads a pickle file, which holds the stored network.
        '''
        try:
            saveNet = pickle.load(open(_pathPickle, "rb" ))
            self.N = saveNet['N']
            self.C = saveNet['C']
            self.label= saveNet['label']
            self.dataset = saveNet['dataset']
            self.datasetPath = saveNet['datasetPath']
            self.cntFinal = saveNet['cntFinal']
            self.wFinal = saveNet['wFinal']
            self.vFinal = saveNet['vFinal']
            self.mstFinal = saveNet['mstFinal']
            self.mrgFinal = saveNet['mrgFinal']
            self.pcaFinal = saveNet['pcaFinal']
            self.wMean = saveNet['wMean']
            self.wStd = saveNet['wStd']
            self.vMean = saveNet['vMean']
            self.vStd = saveNet['vStd']
            self.training = saveNet['training']
            self.pcaExplainedVarMean = saveNet['pcaExplainedVarMean']
            self.pcaExplainedVarStd = saveNet['pcaExplainedVarStd']
            self.pearsonr = saveNet['pearsonr']
            self.pearsonrMean = saveNet['pearsonrMean']
            self.pearsonrStd = saveNet['pearsonrStd']
            self.pearsonrFinal = saveNet['pearsonrFinal']
        except:
            print('Could not load file: ' + str(_pathPickle))



    def writeFile(self, _filename, _data1, _data2=None, _header=None):
        ''' Write data to file.
            If header is given, it is stored before data.
            File looks like:
            [header]
            i j _data[i][j] _data2[i][j]

            Arguments:
                _filename (str): Full path to file.
                _data1 (List): List of data.
                _data2 (List): List of data. Must be of same shape as _data1.
                _header=None (List): 1d list for header information.
        '''

        # Saving to file
        fp = open(_filename, 'w')
        if _header:
            headerStr = ''
            for datum in _header:
                headerStr += str(datum) + '\t'
            headerStr += '\n'
            fp.write(headerStr)

        datumStr = ''
        for i in range(len(_data1)):
            idatum = _data1[i]
            if isinstance(idatum, float) or \
                    isinstance(idatum, int) or \
                    isinstance(idatum, str):
                datumStr += str(i) + '\t0\t' + str(idatum)
                if not _data2 is None:
                    datumStr += '\t' + str(_data2[i])
                datumStr += '\n'
            else:
                for j in range(len(_data1[i])):
                    jdatum = _data1[i][j]
                    datumStr += str(i) + '\t' + str(j) + '\t' + str(jdatum)
                    if not _data2 is None:
                        datumStr += '\t' + str(_data2[i][j])
                    datumStr += '\n'
        fp.write(datumStr)
        fp.close()



    def loadWeights(self, _path):
        ''' This function loads one weights file from _path.

            Arguments:
                _path (str): Path to file to load.
        '''
        # Open file and add to self.training
        print('Loading file ' + str(_path))
        data = []
        with open(_path, newline='') as csvfile:
            r = csv.reader(csvfile, delimiter='\t', quotechar='\"')
            for row in r:
                data.append(row)

        # First row is labels
        for label in data[0]:
            if not label == '':
                self.label.append(label)
        del data[0]
        self.N = len(self.label)
        self.cntFinal = [1]

        # Reset NN
        self.resetNN()

        # Convert str to float
        for cnt1 in range(len(data)):
            for cnt2 in range(len(data[cnt1])):
                data[cnt1][cnt2] = data[cnt1][cnt2].replace(',', '.')
                if data[cnt1][cnt2] == '':
                    del data[cnt1][cnt2]
                else:
                    data[cnt1][cnt2] = float(data[cnt1][cnt2])

        # Is it i, j, weight?
        if len(data) == 3:
            for i, j, weight in data:
                self.w[i][j] = weight
        else:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    self.w[i][j] = data[i][j]

        # Compute results
        self.mstFinal.append(minimum_spanning_tree(self.w))
        self.mrgFinal.append(self.cMrg.computeMrg(self.w)[0])
        self.wFinal.append(self.w)
        self.vFinal.append(self.v)

        # Compute results
        # weights w
        # Numpy seems to have trouble with a list of 2d matrices.
        for i in range(self.N):
            for j in range(self.N):
                row = []
                for k in range(len(self.cntFinal)):
                    row.append(self.wFinal[k][i][j])
                self.wMean[i][j] = np.mean(row)
                self.wStd[i][j] = np.std(row)
        # weights v
        for i in range(self.N):
            row = []
            for k in range(len(self.cntFinal)):
                row.append(self.vFinal[k][i])
            self.vMean[i] = np.mean(row)
            self.vStd[i] = np.std(row)

