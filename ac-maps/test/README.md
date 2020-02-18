# test

tl;dr: WARNING: Do not use Auto Contractive Maps. They do exactly the opposite of what they claim to do.

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Configuration

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/test`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `pathMnist` | Path to mnist dataset as csv file. |
| `pathModel` | Path to model to load. |

MNIST dataset folder must contain `mnist_test.csv` and `mnist_train.csv`. Download from e.g. <https://www.python-course.eu/data/mnist/mnist_train.csv> or <https://www.python-course.eu/data/mnist/mnist_test.csv>.

## Running

The file `test.py` loads a trained net from `pathModel`.
Afterwards, the testing set from the Mnist dataset is compared against the model.
Statistics are printed to std::out.
