# train

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Configuration

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/train`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `folderOut` | Specifies the output folder of all generated files.|
| `pathTemplate` | Path to jinja2 template, which generates a gnuplot script for a heatmap of the weight matrix. |
| `pathMnist` | Path to mnist dataset as csv file. |

MNIST dataset folder must contain `mnist_test.csv` and `mnist_train.csv`. Download from e.g. <https://www.python-course.eu/data/mnist/mnist_train.csv> or <https://www.python-course.eu/data/mnist/mnist_test.csv>.

## Running

The file `train.py` offers four training sets: `random`, `correlated1`, `correlated2`, and `mnist`.
Results will be placed in `folderOut`.

1. create out folder
2. cd ac-maps/train
3. python train.py