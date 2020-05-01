# train

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Configuration

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/datasets`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `folderDatasetOut` | Specifies the output folder of all generated files.|


## Running

The file `createdatasets.py` will place 1000 samples of each  training set `random`, `correlated1`, and `correlated2` in a subfolder in the output folder.
