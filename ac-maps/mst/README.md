# mst

tl;dr: WARNING: Do not use Auto Contractive Maps. They do exactly the opposite of what they claim to do.

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Configuration

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/train`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `folderOut` | Specifies the output folder of all generated files.|
| `pathTemplate` | Path to jinja2 template, which generates a gnuplot script for a heatmap of the weight matrix. |
| `pathWeights` | Path to weights file. First line contains labels. Rest is in format `i j value` or in matrix format. |

## Running

The file `mst.py` loads the specified weights file and creates graphs and weights. They are placed in `folderOut`.
