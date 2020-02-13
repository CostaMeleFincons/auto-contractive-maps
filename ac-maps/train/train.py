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



def main():
    ''' This program trains an Auto Contractive Map.
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
    if not 'pathTemplate' in config:
        raise ValueError('pathTemplate not specified in config file. Exiting.')
    configPathTemplate = str(config['pathTemplate'])
    if not 'pathMnist' in config:
        raise ValueError('pathMnist not specified in config file. Exiting.')
    configPathMnist = str(config['pathMnist'])

    # Length of input vector
    N = 3

    # Contraction parameter
    C = 2

    # Class initialization
    #cAcm = acm(N, C, _dataset='random')
    #cAcm = acm(N, C, _dataset='correlated1')
    #cAcm = acm(N, C, _dataset='correlated2')
    cAcm = acm.Acm(N, C, _dataset='correlated3')
    #cAcm = acm(N, C, _dataset='mnist', _pathMnist=configPathMnist) # for MNIST choose N = 28*28 and C = 100

    # Run training
    cAcm.run(10)

    # Print results
    cAcm.printTree()

    # Print statistics
    cAcm.printStatistics()

    # Draw resulting tree
    #cAcm.draw()

    # Save results to file
    cAcm.save(configFolderOut, configPathTemplate)



if __name__ == "__main__":
    # execute only if run as a script
    main()
