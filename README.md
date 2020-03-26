# Auto Contractive Maps

tl;dr: WARNING: Do not use Auto Contractive Maps. They do exactly the opposite of what they claim to do.

This program implements an Auto Contractive Map algorithm as defined in Buscema, Massimo, et al. "Auto-contractive maps: an artificial adaptive system for data mining. An application to Alzheimer disease." Current Alzheimer Research 5.5 (2008): 481-498.

## Installation

Clone project

```bash
git clone https://github.com/simonreich/auto-contractive-maps.git
cd auto-contractive-maps/ac-maps
```

Optional: Install a local virtual environment for Python.

```bash
virtualenv -p python3 ./
source bin/activate
pip install numpy scipy pandas networkx matplotlib jinja2 pyyaml
```

## Running

There are four project parts. Each part holds its own configuration and README file.

1.  `train`: This program is used to train an AutoCM.
2.  `test`: This program is used to test on the Mnist dataset againt an AutoCM map.
3.  `mrg`: Compute one Maximally Regular Graph (MRG).
4.  `helper`: Helper functions and classes.
