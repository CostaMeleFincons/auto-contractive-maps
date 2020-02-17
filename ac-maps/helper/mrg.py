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



import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree



class Mrg:
    ''' This class implements a Maximally Regular Graph (MRG) algorithm.
    '''

    def __init__(self):
        ''' Initialization function.
        '''



    def computeMrg(self, _w):
        ''' This function computes a Maximally Regular Graph (MRG) given a weight matrix.

            Arguments:
                _w (np.array): weight matrix of shape (N, N).
        '''
        # Compute mst
        mst = minimum_spanning_tree(_w)
        mst = np.array(mst.toarray().astype(float))

        # Get list of [[i, j, weight]], which are not in mst
        weightlist = []
        for i in range(len(mst)):
            for j in range(i+1, len(mst[i])):
                if not i == j:
                    if mst[i][j] == 0 and mst[j][i] == 0:
                        if _w[i][j] > _w[j][i]:
                            weightlist.append([i, j, _w[i][j]])
                        else:
                            weightlist.append([i, j, _w[j][i]])

        # Sort list by weight
        weightlist.sort(key=lambda x: x[2])

        # Get current H Score
        hOld = self.getH(mst)

        # Add weights and check new H score
        for [i, j, weight] in weightlist:
            # Add weight to graph
            mst[i][j] = _w[i][j]
            mst[j][i] = _w[j][i]

            # Get new H score
            hNew = self.getH(mst)

            # Discard change, if new H score is smaller
            if hNew < hOld:
                mst[i][j] = 0
                mst[j][i] = 0
            else:
                hOld = hNew

        # Return new graph
        return mst, hOld



    def getH(self, _w):
        ''' Computes the H score of one given graph.

            Arguments:
                _w (np.array): weight matrix of shape (N, N).

            Returns:
                int. H Score
        '''
        # prune
        t = np.copy(_w)
        prunelist = self.prune(t, _prunelist=[])

        # Compute H score
        # A = number of edges 
        t = np.copy(_w)
        A = self.getNrEdges(t, _binarize=True)[1]

        # mu = number of edges / number of prune iterations
        mu = A / len(prunelist)

        # S = list of different gradients
        S = [x[0] for x in prunelist]
        S = np.unique(S)

        # P number of different gradients
        P = len(S)

        # phi = 1/P * sum(S)
        phi = 1/P * np.sum(S)

        # H score
        H = (mu*phi - 1) / A

        # return H
        return H



    def getNrEdges(self, _w, _binarize=False):
        ''' Computes the number of edges in a graph.

            Arguments:
                _w (np.array): weight matrix of shape (N, N) with binary input. If _w not binary, use _binarize=True.
                _binarize (bool): will replace values !=0 with 1.

            Returns:
                np.array. Array holding edges per node.
                int. Total number of edges np.sum(Array holding edges per node).
        '''
        # Binarize
        if _binarize:
            for i in range(len(_w)):
                _w[i][i] = 0
                for j in range(len(_w[i])):
                    if not _w[i][j] == 0:
                        _w[i][j] = 1.0
                        _w[j][i] = 1.0

        # Get number of edges per Node
        nrEdges = np.sum(_w, axis=0)

        # Get total number of edges
        nrEdgesTotal = np.sum(nrEdges)/2

        return nrEdges, nrEdgesTotal



    def prune(self, _w, _prunelist=[]):
        ''' This function prunes iteratively a graph.

            Arguments:
                _w (np.array): weight matrix of shape (N, N).
                _prunelist ([int, int, int]): Prune list [Gradient nodes removed, Edges removed, Nodes removed].
        '''
        # Get number of edges
        if len(_prunelist) == 0:
            nrEdges, nrEdgesTotal = self.getNrEdges(_w, _binarize=True)
        else:
            nrEdges, nrEdgesTotal = self.getNrEdges(_w, _binarize=False)

        # Get total number of nodes
        nrNodesTotal = np.where(nrEdges>0)[0].size

        # Find minimum in number of edges
        nrEdgesMin = np.copy(nrEdges)
        nrEdgesMin[nrEdgesMin==0] = np.inf
        nrEdgesMin = np.where(nrEdgesMin==nrEdgesMin.min())
        #print('nrEdges ', nrEdges)
        #print('nrEdgesTotal ', nrEdgesTotal)
        #print('nrNodesTotal ' , nrNodesTotal)
        #print('nrEdgesMin ', nrEdgesMin)

        # Remove minimum from graph
        for index in np.nditer(nrEdgesMin):
            _w[index] = 0
            _w.T[index] = 0

        # Get new number of edges per Node
        nrEdgesNew = np.sum(_w, axis=0)

        # Get new total number of edges
        nrEdgesTotalNew = np.sum(nrEdgesNew)/2

        # Get new total number of nodes
        nrNodesTotalNew = np.where(nrEdgesNew>0)[0].size

        # Append to prune list
        edgesRemoved = int(nrEdgesTotal-nrEdgesTotalNew)
        nodesRemoved = int(nrNodesTotal-nrNodesTotalNew)
        gradient = 1
        if len(_prunelist) > 0:
            gradient = abs(_prunelist[-1][2]-nodesRemoved)
        _prunelist.append([gradient, edgesRemoved, nodesRemoved])

        # Continue pruning
        if np.sum(_w) > 0:
            _prunelist = self.prune(np.copy(_w), _prunelist)

        return _prunelist


