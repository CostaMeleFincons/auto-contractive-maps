# Maximally Regular Graph (MRG)

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Configuration

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/mrg`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `folderOut` | Specifies the output folder of all generated files.|
| `pathModel` | Path to model to load. |

MNIST dataset folder must contain `mnist_test.csv` and `mnist_train.csv`. Download from e.g. <https://www.python-course.eu/data/mnist/mnist_train.csv> or <https://www.python-course.eu/data/mnist/mnist_test.csv>.

## Running

The file `mrg.py` loads a trained net from `pathModel`.
Afterwards, the MRG is computed.
This is meant as a testing program for the mrg class.
Results are placed in `folderOut`.
