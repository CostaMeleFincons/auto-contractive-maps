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

First, please have a look at the file `config.yaml` in `auto-contractive-maps/ac-maps/train`. There parameters are:

| Parameter | Description |
|-----------|-------------|
| `folderOut` | Specifies the output folder of all generated files.|
| `pathTemplate` | Path to jinja2 template, which generates a gnuplot script for a heatmap of the weight matrix. |

To run the program use

```bash
cd auto-contractive-maps/ac-maps/train
python ./train.py
```

## Experiments

The file `train.py` offers two training sets: `cAcm.createTrainingRandom()` and `cAcm.createTrainingCorrelated()`.

The first set, `cAcm.createTrainingRandom()`, creates 1000 training samples of length 10, which are drawn randomly from a uniform distribution.

The second set, `cAcm.createTrainingCorrelated()`, creates 1000 training samples of length 10, where each entry correlates heavily with each other. The entries of each vector in detail:

1.  The first entry is generated randomly from a uniform distribution `[0, 1]`, let it be `R1` (`v[0] = np.random.rand(1)[0]`).
2.  This entry is double of `R1` (`v[1] = v[0]*2`).
3.  This entry is `R1`, but a small offset of `0.1` is added (`v[2] = v[0]+0.1`)
4.  This entry is `R1` squared (`v[3] = v[0]*v[0]`).
5.  This entry is `R1 squared and doubled (`v[4] = v[0]*v[0]*2`).
6.  This entry is `R1 squared and tripled (`v[5] = v[0]*v[0]*3`).
7.  This entry is drawn randomly from a uniform distribution `[0.9, 1]` (`v[6] = np.random.rand(1)[0]*0.1 + 0.9`)
8.  This entry is drawn randomly from a uniform distribution `[0.9, 1]` (`v[7] = np.random.rand(1)[0]*0.1 + 0.9`)
9.  This entry is drawn randomly from a uniform distribution `[0.9, 1]` (`v[8] = np.random.rand(1)[0]*0.1 + 0.9`)
10.  This entry is drawn randomly from a uniform distribution `[0.9, 1]` (`v[9] = np.random.rand(1)[0]*0.1 + 0.9`)

### Results for first set

Expected outcome: No clustering from the first set.

Actual outcome: Clusters build randomly.

The algorithm creates a graph, which always includes all nodes. This suggests correlations, even though there are none.
The weights for all connections are usually in the range [1.1, 1.6], meaning there is no large difference.

### Outcome for the first set

Expected outcome: Two clusters from the second set. Clusters should be linked between one of 2,3 and 7, 8, 9, 10 (2 to 7, 8, 9, 10 or 3 to 7, 8, 9, 10).

Actual outcome: All entries are centered around entry 4: `R1` squared (`v[3] = v[0]*v[0]`).

The graph is computed using minimum spanning tree (MST). This algorithm tries to minimize the total cost of edges, while still visiting all edges. It will always favor small input values, which will result in small weights. Replacing any entry with an even smaller number, e.g. `R1^3` or `1e-3`, will center the graph around this node.

## Discussion

Auto contractive maps fail to show correlations in heavily correlated data. Instead, it suggests either random connections between data points, or connections, which correlate to small input values.

Furthermore, the stopping criteria in the paper says that one should stop learning, if the output neurons are `0`. First, this is an utterly random criteria (even though the formulas are created such that the weight change will also drop to 0, if the output neurons turn to zero). The number of training samples used depends on the hyperparameter C.

For the first time steps (assuming first layer weights `\vec{v}_i << 1` and second layer weights `W_{ij} << 1`) one can show that the weight change `\Delta W_{ij}` is `\Delta W_{ij} \approx N/2 x m^s_i x m^s_j`, where `m_s` is the input vector and `N` its length. Afterwards, this is summed up and normalized again, meaning that iteratively something like a correlation matrix is build (at least for the first iterations. However, learning slows down after the first few iterations.). 
The graph extraction algorithm, minimum spanning tree, than selects the edges according to their lowest values (corresponding to low "correlation") and presents it to the user. This is most likely the opposite of what the user actually wants and expects.
