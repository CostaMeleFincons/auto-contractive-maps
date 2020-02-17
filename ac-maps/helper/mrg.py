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



    def computeMrg(self, w):
        ''' This function computes a Maximally Regular Graph (MRG) given a weight matrix.

            Arguments:
                w (np.array): weight matrix of shape (N, N).
        '''

        # Compute mst
        mst = minimum_spanning_tree(w)
        mst = np.array(mst.toarray().astype(float))

        # prune
        prunelist = self.prune(mst)

        print(prunelist)



    def prune(self, _w, _prunelist=[]):
        ''' This function prunes iteratively a graph.

            Arguments:
                _w (np.array): weight matrix of shape (N, N).
                _prunelist ([int, int, int]): Prune list [Gradient nodes removed, Edges removed, Nodes removed].
        '''
        print('In ', _w)


        for i in range(len(_w)):
            for j in range(len(_w[i])):
                if _w[i][j] > 0:
                    _w[i][j] = 1.0
                    _w[j][i] = 1.0

        # Get number of edges per Node
        nrEdges = np.sum(_w, axis=0)

        # Get total number of edges
        nrEdgesTotal = np.sum(nrEdges)/2

        # Get total number of nodes
        nrNodesTotal = np.where(nrEdges>0)[0].size

        # Find minimum in number of edges
        nrEdgesMin = np.where(nrEdges==nrEdges.min())
        print('nrEdgesMin ', nrEdgesMin)

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
        edgesRemoved = nrEdgesTotalNew-nrEdgesTotal
        nodesRemoved = nrNodesTotalNew-nrNodesTotal
        gradient = 1
        if len(_prunelist) > 0:
            gradient = abs(_prunelist[-1][2]-nodesRemoved)
        _prunelist.append([gradient, edgesRemoved, nodesRemoved])
        input("Press Enter to continue...")

        # Continue pruning
        if np.sum(_w) > 0:
            _prunelist = self.prune(np.copy(_w), _prunelist)

        return _prunelist


