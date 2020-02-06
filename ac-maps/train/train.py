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



"""
Create ac-maps and test.
"""



import numpy as np
import os
from copy import deepcopy
from pathlib import Path

import sys
sys.path.insert(0, '../helper')
import filor



def main():
    # Input data length
    N = 10

    # Contraction parameter
    C = 2

    # First layer
    v = np.full((1, N), 0.01, dtype=float)[0]

    # Hidden layer
    w = np.full((N, N), 0.01, dtype=float)

    for cnt in range(10):
        vIn = np.random.rand(1, N)[0]

        # 1. Signal In to Hidden
        vHidden =  np.zeros((1, N), dtype=float)[0]
        for i in range(N):
            vHidden[i] = vIn[i] * (1 - v[i]/C)

        # 2. Adapt weights In to Hidden (v)
        # The formula is actualle m_s * (1 - (v/C)^2)
        for i in range(N):
            v[i] = (vIn[i] - vHidden[i]) * (1 - (v[i]/C))

        # 3. Signal Hidden to Out



        print(v)





def save(_fileDatabase, _databaseProcessed, _database):
    ''' Saves the database to file
    '''
    # Header
    header = ['Filename', 'label:0nogm;1gmpresent;2unknown/undecided;3sleep;4fuzzy', 'comment']

    # All items, which do not have None as label
    databaseSave = [x for x in (_databaseProcessed+_database) if x[1] is not None]

    # Save
    cFilor = filor.Filor(True)
    cFilor.writeFile(_fileDatabase, databaseSave, header, True)



if __name__ == "__main__":
    # execute only if run as a script
    main()
