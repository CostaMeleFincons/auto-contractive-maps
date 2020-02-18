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
import random
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import jinja2

import sys
sys.path.insert(0, '../helper')
import mrg


class Acm:
    ''' This class implements an Auto Contractive Map.
    '''

    def __init__(self, _inputLength, _contraction, _dataset, _pathMnist=None):
        ''' Initialization function.

            Arguments:
                _inputLength (int): Length of input vector
                _contraction (float): Contraction parameter, _contraction>1.
                _dataset (str): Name of dataset to use: 'random', 'correlated1', 'correlated2', 'correlated3', or 'mnist'.
                _pathMnist=None (str): Path to MNIST dataset as csv file. Must contain mnist_test.csv and mnist_train.csv.
        '''

        # Length of input vector
        self.N = _inputLength

        # Contraction parameter
        self.C = _contraction

        # Path to MNIST dataset
        self.pathMnist = _pathMnist

        # Dataset to use
        self.dataset = _dataset
        if _dataset == 'mnist':
            self.loadMnist()

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

        # Reset
        self.resetNN()

        # MRG class object
        self.cMrg = mrg.Mrg()



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



    def runOnce(self, _mIn):
        ''' This function performs one run of training using _mIn as input vector.

            Arguments:
                _mIn (np.array(dtype=float)): Input vector.
        '''

        # 0. Normalize input to be [0, 1]
        mIn = np.interp(_mIn, (_mIn.min(), _mIn.max()), (0, 1))
        assert np.amin(mIn) >= 0, 'Training sample holds data <0: ' + str(mIn)
        assert np.amax(mIn) <= 1, 'Training sample holds data >1: ' + str(mIn)

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



    def createTrainingRandom(self):
        ''' This function creates 1000 training samples.

            The each vector is drawn randomly from a uniform distribution, meaning there is no correlation.
        '''
        # Create random training data
        self.training = np.random.rand(1000, self.N)

        # Create labels
        self.label = []
        for j in range(self.N):
            self.label.append('R' + str(j))



    def createTrainingCorrelated1(self):
        ''' This function creates 1000 training samples.

            The each vector is made up as follows:
            ['R0', 'f1(R0)', 'f2(R0)', 'f3(R0)', 'f4(R0)', 'f5(R0)', 'R6', 'R7', 'R8', 'R9']
        '''

        if self.N < 10:
            raise ValueError('For createTrainingCorrelated1 an input vector size of at least 10 is needed.')

        # Create labels
        self.label = ['R0', 'f1(R0)', 'f2(R0)', 'f3(R0)', 'f4(R0)', 'f5(R0)', 'R6', 'R7', 'R8', 'R9'] 

        # Create correlated training data
        self.training = []
        for i in range(1000):
            v = np.zeros(self.N, dtype=float)
            v[0] = np.random.rand(1)[0]
            v[1] = v[0]*2
            v[2] = v[0]+0.1
            v[3] = v[0]*v[0]
            v[4] = v[0]*v[0]*2
            v[5] = v[0]*v[0]*3
            v[6] = np.random.rand(1)[0]
            v[7] = np.random.rand(1)[0]
            v[8] = np.random.rand(1)[0]
            v[9] = np.random.rand(1)[0]

            self.training.append(v)



    def createTrainingCorrelated2(self):
        ''' This function creates 1000 training samples.

            The each vector is made up as follows:
            ['R0', 'R1(R0)', 'R2(R0)', 'R3(R0)', 'R4(R0)', 'R5(R0)', 'R6', 'R7(R6)', 'R8', 'R9(R8)']
        '''

        if self.N < 10:
            raise ValueError('For createTrainingCorrelated2 an input vector size of at least 10 is needed.')

        # Create labels
        self.label = ['R0', 'R1(R0)', 'R2(R0)', 'R3(R0)', 'R4(R0)', 'R5(R0)', 'R6', 'R7(R6)', 'R8', 'R9(R8)'] 

        # Create correlated training data
        self.training = []
        for i in range(1000):
            v = np.zeros(self.N, dtype=float)
            v[0] = np.random.rand(1)[0]
            v[1] = np.random.rand(1)[0]*0.1
            v[2] = np.random.rand(1)[0]*0.1
            v[3] = np.random.rand(1)[0]*0.1
            v[4] = np.random.rand(1)[0]*0.1
            v[5] = np.random.rand(1)[0]*0.1
            if v[0] > 0.5:
                v[1] += 0.9
                v[2] += 0.9
                v[3] += 0.9
                v[4] += 0.9
                v[5] += 0.9
            v[6] = np.random.rand(1)[0]*2-1
            v[7] = -1*v[6]
            v[8] = np.random.rand(1)[0]*2-1
            v[9] = -1*v[8]

            self.training.append(v)



    def createTrainingCorrelated3(self):
        ''' This function creates 1000 training samples.

            The each vector is made up as follows:
            ['[0.15, 0.25]', '[0.35, 0.45]', '[0.95, 1.05]']
        '''

        if self.N < 3:
            raise ValueError('For createTrainingCorrelated3 an input vector size of at least 3 is needed.')

        # Create labels
        self.label = ['0.2', '0.4', '1']

        # Create correlated training data
        self.training = []
        for i in range(1000):
            v = np.zeros(self.N, dtype=float)
            v[0] = np.random.rand(1)[0]*0.1-0.05 + 0.2
            v[1] = np.random.rand(1)[0]*0.1-0.05 + 0.4
            v[2] = np.random.rand(1)[0]*0.1-0.05 + 1.0

            self.training.append(v)



    def createTrainingMnist(self):
        ''' This function loads the mnist dataset as training data.

            As input parameter it stacks the values of all ones. All other numbers are ignored.
            1000 Input vectors are created.
        '''

        if self.N < 28*28:
            raise ValueError('For createTrainingMnist an input vector size of at least 28*28 is needed.')

        # Create labels
        #self.label = ['R0', 'R1(R0)', 'R2(R0)', 'R3(R0)', 'R4(R0)', 'R5(R0)', 'R6', 'R7(R6)', 'R8', 'R9(R8)'] 

        # Extract numbers 1 and shuffle
        data1 = [x[1:] for x in self.mnistTrain if x[0] == 1]
        random.shuffle(data1)

        self.training = np.array(data1[0:1000], dtype=float)



    def loadMnist(self):
        ''' This function loads the mnist dataset from file.
        '''
        # Filename checks
        filenameTraining = os.path.join(self.pathMnist, 'mnist_train.csv')
        filenameTest = os.path.join(self.pathMnist, 'mnist_test.csv')
        if not os.path.isdir(self.pathMnist):
            raise ValueError('Folder not found: ' + str(self.pathMnist))
        if not os.path.isfile(filenameTraining):
            raise ValueError('File not found: ' + str(filenameTraining))
        if not os.path.isfile(filenameTest):
            raise ValueError('File not found: ' + str(filenameTest))

        # Load data
        self.mnistTrain = np.loadtxt(filenameTraining, delimiter=",")
        self.mnistTest = np.loadtxt(filenameTest, delimiter=",") 



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
            if self.dataset == 'random':
                self.createTrainingRandom()
            elif self.dataset == 'correlated1':
                self.createTrainingCorrelated1()
            elif self.dataset == 'correlated2':
                self.createTrainingCorrelated2()
            elif self.dataset == 'correlated3':
                self.createTrainingCorrelated3()
            elif self.dataset == 'mnist':
                self.createTrainingMnist()
            else:
                raise ValueError('No dataset specified.')

            # Runtime counter
            cnt = 0
            successfull = True
            for x in self.training:
                # For random data, the self.mOut will sometimes run away and will create a buffer overflow
                if any(np.greater(self.mOut, 1e+10)) is True or \
                        any(np.less(self.mOut, -1e+10)) is True:
                    successfull = False
                    cntNr -= 1
                    break

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



    def printTree(self):
        ''' This function prints the last results of self.run().
        '''

        print('Total number of training samples: ' + str(self.cntFinal[-1]) + '\n')

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

        print('Mean weights w: ' + str(self.wMean))
        print('Std weights w: ' + str(self.wStd))

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



    def testMnist(self):
        ''' This function tests learned weights against the MNIST testing set.
        '''

        if self.N < 28*28:
            raise ValueError('For testMnist an input vector size of at least 28*28 is needed.')

        # Extract data as [[input vector], label, prediction=None]
        data = [[np.array(x[1:]), int(x[0]), None] for x in self.mnistTest]

        for cnt in range(len(data)):
            # Perform forward pass
            x = data[cnt]

            # 0. Normalize input to be [0, 1]
            mIn = np.interp(x[0], (x[0].min(), x[0].max()), (0, 1))
            assert np.amin(mIn) >= 0, 'Training sample holds data <0: ' + str(mIn)
            assert np.amax(mIn) <= 1, 'Training sample holds data >1: ' + str(mIn)

            # 1. v
            mHidden = np.multiply(mIn, self.vFinal[-1])

            # 2. w
            mOut = np.matmul(self.wFinal[-1], mHidden)

            # Sum up results
            if np.sum(mOut) > 8000000:
                data[cnt][2] = 1
            else:
                data[cnt][2] = 0


        # Compute statistics

        # Evaluate 1 vs all
        # This treats label 1 as 1, and label 0, 2, 3, 4, ... as 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for [vector, label, labelPrediction] in data:
            # Get label
            if label == 1:
                pass
            else:
                label = 0
            if labelPrediction == 1:
                pass
            else:
                labelPrediction = 0
 
            if label == 1:
                if labelPrediction == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if labelPrediction == 1:
                    fp += 1
                else:
                    tn += 1
 
        # Confusion matrix
        print('\n\n1 vs all results:\n  tp fp {0:6.0f} {1:6.0f}\n  fn tn {2:6.0f} {3:6.0f} '.format(
                tp, fp,
                fn, tn))
 
        # Precision
        precision = 0
        if (tp + fp) == 0:
            print('\n  Precision: {0}'.format('Division by zero'))
        else:
            precision = tp/(tp + fp)
            print('\n  Precision: {0:2.4f}'.format(precision))
 
        # Recall
        recall = 0
        if (tp + fp) == 0:
            print('     Recall: {0}'.format('Division by zero'))
        else:
            recall = tp/(tp + fn)
            print('     Recall: {0:2.4f}'.format(recall))
 
        # F1-Score
        f1score = 0
        if (precision+recall) == 0:
            print('   F1-Score: {0}'.format('Division by zero'))
        else:
            f1score = 2*precision*recall/(precision+recall)
            print('   F1-Score: {0:2.4f}'.format(f1score))
 
        # Accuracy
        if (tp+fp+fn+tn) == 0:
            print('   Accuracy: {0}'.format('Division by zero'))
        else:
            acc = (tp+tn)/(tp+fp+fn+tn)
            print('   Accuracy: {0:2.4f}'.format(acc))



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
        # Filenames
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = str(date) + '_dataset-' + str(self.dataset) + '_nrruns-' + str(len(self.cntFinal))
        filenameWeights = os.path.join(_folderOut, filename + '_weights.txt')
        filenameWeightsPlot = os.path.join(_folderOut, filename + '_weights.plot')
        filenameWeightsPng = os.path.join(_folderOut, filename + '_weights.png')
        filenameWeightsTex = os.path.join(_folderOut, filename + '_weights.tex')
        filenameWeightsMean = os.path.join(_folderOut, filename + '_weightsmean.txt')
        filenameWeightsMeanPlot = os.path.join(_folderOut, filename + '_weightsmean.plot')
        filenameWeightsMeanPng = os.path.join(_folderOut, filename + '_weightsmean.png')
        filenameWeightsMeanTex = os.path.join(_folderOut, filename + '_weightsmean.tex')
        filenameGraphMst = os.path.join(_folderOut, filename + '_mst.png')
        filenameGraphMstMean = os.path.join(_folderOut, filename + '_mstmean.png')
        filenameGraphMrg = os.path.join(_folderOut, filename + '_mrg.png')
        filenameGraphMrgMean = os.path.join(_folderOut, filename + '_mrgmean.png')
        filenamePickle = os.path.join(_folderOut, filename + '_net.p')

        # Save weights of last run
        self.writeFile(filenameWeights, np.around(self.w, decimals=3))

        # Create gnuplot scripts from jinja2 template of last run
        templateLoader = jinja2.FileSystemLoader(searchpath=_folderOut)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenameWeightsPng, 
                filenameWeightsTex=filenameWeightsTex,
                filenameWeights=os.path.basename(filenameWeights),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange=str(int(np.amin(self.w))) + ':' + str(int(np.amax(self.w))+1))

        fp = open(filenameWeightsPlot, 'w')
        fp.write(script)
        fp.close()

        # Save weights mean
        self.writeFile(filenameWeightsMean, np.around(self.wMean, decimals=3))

        # Create gnuplot scripts from jinja2 template mean
        templateLoader = jinja2.FileSystemLoader(searchpath=_folderOut)
        templateEnv = jinja2.Environment(loader=templateLoader)
        template = templateEnv.get_template('weights.plot.jinja2')
        script = template.render(filenameWeightsPng=filenameWeightsMeanPng, 
                filenameWeightsTex=filenameWeightsMeanTex,
                filenameWeights=os.path.basename(filenameWeightsMean),
                range=str(-0.5) + ':' + str(self.N-1+0.5),
                cbrange=str(int(np.amin(self.wMean)*10)/10) + ':' + str(int(np.amax(self.wMean)*10)/10+0.1))

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

        # Save all weights
        saveNet = {}
        saveNet['N'] = self.N
        saveNet['C'] = self.C
        saveNet['label'] = self.label
        saveNet['dataset'] = self.dataset
        saveNet['pathMnist'] = self.pathMnist
        saveNet['cntFinal'] = self.cntFinal
        saveNet['wFinal'] = self.wFinal
        saveNet['vFinal'] = self.vFinal
        saveNet['mstFinal'] = self.mstFinal
        saveNet['mrgFinal'] = self.mrgFinal
        saveNet['wMean'] = self.wMean
        saveNet['wStd'] = self.wStd
        saveNet['vMean'] = self.vMean
        saveNet['vStd'] = self.vStd
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
            self.pathMnist = saveNet['pathMnist']
            self.cntFinal = saveNet['cntFinal']
            self.wFinal = saveNet['wFinal']
            self.vFinal = saveNet['vFinal']
            self.mstFinal = saveNet['mstFinal']
            self.mrgFinal = saveNet['mrgFinal']
            self.wMean = saveNet['wMean']
            self.wStd = saveNet['wStd']
            self.vMean = saveNet['vMean']
            self.vStd = saveNet['vStd']
        except:
            print('Could not load file: ' + str(_pathPickle))



    def writeFile(self, _filename, _data, _header=None, _debug=False):
        ''' Write data to file.
            If header is given, it is stored before data.

            Arguments:
                _filename (str): Full path to file.
                _data (List): List of data.
                _header=None (List): 1d list for header information.
                _debug=False (bool): Turn debug output for this class on/off. Can also be set by self.debug=True/False.
        '''
        self.debug = _debug

        # Saving to file
        if self.debug: print('Saving to file ' + str(_filename))
        fp = open(_filename, 'w')
        if _header:
            headerStr = ''
            for datum in _header:
                headerStr += str(datum) + '\t'
            headerStr += '\n'
            fp.write(headerStr)
        for datum in _data:
            datumStr = ''
            cnt = 0
            for value in datum:
                datumStr += str(value)
                if cnt < len(datum)-1:
                    datumStr += '\t'
                cnt += 1
            datumStr += '\n'
            fp.write(datumStr)
        fp.close()
