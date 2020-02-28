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
    if not 'folderDatasetOut' in config:
        raise ValueError('folderOut not specified in config file. Exiting.')
    configFolderDatasetOut = str(config['folderDatasetOut'])

    # Number of samples
    numSamples = 1000

    # Class initialization
    for dataset in ['random', 'correlated1', 'correlated2', 'correlated3']:
        cAcm = acm.Acm(10, 1, _dataset=dataset)
        cAcm.saveDataset(configFolderDatasetOut, _nr=1000)



if __name__ == "__main__":
    # execute only if run as a script
    main()
