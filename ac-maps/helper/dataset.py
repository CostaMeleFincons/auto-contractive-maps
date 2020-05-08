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


import csv
import os
import numpy as np

import sys
sys.path.insert(0, '../helper')



class Dataset:
    ''' This class implements dataset for acm.
    '''
    def __init__(self):
        ''' Initialization function.
        '''
        # Is Mnist loaded?
        self.isMnistLoaded = False

        # Mnist dataset
        self.mnistTrain = None
        self.mnistTest = None

        return



    def createTrainingRandom(self, _N):
        ''' This function creates 1000 training samples.

            The each vector is drawn randomly from a uniform distribution, meaning there is no correlation.

            Arguments:
                _N (int): Number of parameters
        '''
        # Create random training data
        training = np.random.rand(1000, _N)

        # Create labels
        label = []
        for j in range(_N):
            label.append('m' + str(j))

        return training, label



    def createTrainingCorrelated1(self):
        ''' This function creates 1000 training samples.

            The each vector is made up as follows:
            ['R0', 'f1(R0)', 'f2(R0)', 'f3(R0)', 'f4(R0)', 'f5(R0)', 'R6', 'R7', 'R8', 'R9']
        '''
        # Create labels
        label = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'] 

        # Create correlated training data
        training = []
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

            training.append(v)

        return training, label



    def createTrainingCorrelated2(self):
        ''' This function creates 1000 training samples.

            The each vector is made up as follows:
            ['R0', 'R1(R0)', 'R2(R0)', 'R3(R0)', 'R4(R0)', 'R5(R0)', 'R6', 'R7(R6)', 'R8', 'R9(R8)']
        '''
        # Create labels
        label = ['m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'] 

        # Create correlated training data
        training = []
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

            training.append(v)

        return training, label



    def loadMnist(self):
        ''' This function loads the mnist dataset from file.
        '''
        # Filename checks
        filenameTraining = os.path.join(self.pathDataset, 'mnist_train.csv')
        filenameTest = os.path.join(self.pathDataset, 'mnist_test.csv')
        if not os.path.isdir(self.pathDataset):
            raise ValueError('Folder not found: ' + str(self.pathDataset))
        if not os.path.isfile(filenameTraining):
            raise ValueError('File not found: ' + str(filenameTraining))
        if not os.path.isfile(filenameTest):
            raise ValueError('File not found: ' + str(filenameTest))

        # Load data
        self.mnistTrain = np.loadtxt(filenameTraining, delimiter=",")
        self.mnistTest = np.loadtxt(filenameTest, delimiter=",") 

        self.isMnistLoaded = True



    def createTrainingMnist(self):
        ''' This function loads the mnist dataset as training data.

            As input parameter it stacks the values of all ones. All other numbers are ignored.
            1000 Input vectors are created.
        '''
        # Load Mnist
        if self.isMnistLoaded == False:
            self.loadMnist()

        # Extract numbers 1 and shuffle
        data1 = [x[1:] for x in self.mnistTrain if x[0] == 1]
        random.shuffle(data1)

        training = np.array(data1[0:1000], dtype=float)

        return training



    def saveDataset(self, _folder, _dataset, _training, _label):
        ''' This function saves the used dataset to file.

            Arguments:
                _folder (str): All output files will be placed in this folder.
                _dataset (str): Name of dataset. Used in filename
                _training (List[List[float]]): Training data to store.
                _label (List[str]): Labels of dataset.
        '''
        # Filenames
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = str(date) + '_dataset-' + str(_dataset)

        # Create folder
        folder = os.path.join(_folder, _dataset)
        if not os.path.isdir(folder):
            try:
                os.mkdir(folder)
            except OSError:
                print ("Creation of the directory %s failed" % folder)


        filenameOut = os.path.join(folder, filename + '.nnv')
     
        # Store to file
        with open(filenameOut, 'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter='\t', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            w.writerow(_label)
            for row in _training:
                w.writerow(row)



    def createAndSaveDataset(self, _folder, _dataset, _label, _nr=1000):
        ''' This function saves the used dataset to file.

            Arguments:
                _folder (str): All output files will be placed in this folder.
                _dataset (str): Name of dataset. Used in filename
                _label (List[str]): Labels of dataset.
                _nr=1000: How many datasets should be created. If set to None, current dataset is stored.
        '''
        # Filenames
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = str(date) + '_dataset-' + str(_dataset)

        # Create folder
        folder = os.path.join(_folder, _dataset)
        if not os.path.isdir(folder):
            try:
                os.mkdir(folder)
            except OSError:
                print ("Creation of the directory %s failed" % folder)

        for cnt in range(abs(_nr)):
            filenameOut = filename + '_nr-' + str(cnt)
            filenameOut = os.path.join(folder, filenameOut + '.nnv')

            # Create data
            if _dataset == 'random':
                trainig, label = self.createTrainingRandom()
            elif _dataset == 'correlated1':
                trainig, label = self.createTrainingCorrelated1()
            elif _dataset == 'correlated2':
                trainig, label = self.createTrainingCorrelated2()
            elif _dataset == 'correlated3':
                trainig, label = self.createTrainingCorrelated3()
            elif _dataset == 'mnist':
                trainig, label = self.createTrainingMnist()
            else:
                raise ValueError('No dataset specified.')

            # Store to file
            with open(filenameOut, 'w', newline='') as csvfile:
                w = csv.writer(csvfile, delimiter='\t', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                w.writerow(label)
                for row in training:
                   w.writerow(row)



    def loadDataset(self, _folder, _nr=None):
        ''' This function loads one dataset from file.

            Arguments:
                _folder (str): Folder to load files from.
                _nr=None: If the dataset has a number '_nr-xxx', number xxx is loaded.
        '''
        # Load all files from dir
        filelist = []
        for filename in os.listdir(_folder):
            filename = os.path.join(_folder, filename)
            filelist.append(filename)
        filelist = sorted(filelist)

        # Search for file
        filenameLoad = None
        for filename in filelist:
            foundDataset = False
            foundNr = False
            dataset = None
            nr = None
            splits = os.path.splitext(os.path.basename(filename))[0].split('_')
            for split in splits:
                if split[0:8] == 'dataset-':
                    dataset = split[8:]
                    foundDataset = True
                if split[0:3] == 'nr-':
                    if int(split[3:]) == _nr:
                        nr = int(split[3:])
                        foundNr = True

            if _nr is None:
                if foundDataset:
                    filenameLoad = filename
                    break
            else:
                if foundDataset and foundNr:
                    filenameLoad = filename
                    break

        # Open file and add to training
        #print('Loading file ' + str(filenameLoad))
        data = []
        with open(filenameLoad, newline='') as csvfile:
            r = csv.reader(csvfile, delimiter='\t', quotechar='\"')
            for row in r:
                data.append(row)

        label = data[0]
        training = np.array(data[1:], dtype=float)
        return training, label



    def loadFile(self, _path):
        ''' This function loads one dataset from file.

            Arguments:
                _folder (str): Folder to load files from.
                _nr=None: If the dataset has a number '_nr-xxx', number xxx is loaded.
        '''
        # Open file and add to training
        #print('Loading file ' + str(filenameLoad))
        data = []
        with open(_path, newline='') as csvfile:
            r = csv.reader(csvfile, delimiter='\t', quotechar='\"')
            for row in r:
                data.append(row)

        label = data[0]
        training = np.array(data[1:], dtype=float)
        return training, label
