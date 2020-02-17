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



import yaml
import sys
sys.path.insert(0, '../helper')
import acm
import mrg



def main():
    ''' This program computes a Maximally Regular Graph (MRG) given a weight matrix.
    '''

    # Read yaml config file
    filename = './config.yaml'
    config = ''
    try:  
        with open(str(filename), 'r') as fp:  
            for cnt, line in enumerate(fp):
                config += str(line)
    finally:  
        fp.close()
    try:
        config = yaml.safe_load(config)
    except yaml.YAMLError as exc:
        print(exc)
        raise
    if not 'folderOut' in config:
        raise ValueError('folderOut not specified in config file. Exiting.')
    configFolderOut = str(config['folderOut'])
    if not 'pathModel' in config:
        raise ValueError('pathModel not specified in config file. Exiting.')
    configPathModel = str(config['pathModel'])

    # Length of input vector
    N = 10

    # Contraction parameter
    C = 50

    # Class initialization
    cAcm = acm.Acm(N, C, _dataset='correlated2')

    # Load acm
    cAcm.load(configPathModel)

    # Pass weights to mgr
    cMrg = mrg.Mrg()
    cMrg.computeMrg(cAcm.wMean)





if __name__ == "__main__":
    # execute only if run as a script
    main()
