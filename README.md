# TRKG-Miner: Relaxed Temporal Knowledge Graph Miner

TRKG-Miner is a framework for link forecasting on knowledge graphs. It mines both cyclic and acyclic rules and uses them to predict links in the knowledge graph.

Note that the task of link forecasting is quite different from link prediction. While many deep learning based approaches have worked successfully for the latter task, the former remains a challenging problem. TRKG-Miner is a rule-based approach that shows significant improvement over the state-of-the-art approaches for link forecasting.

### Due credit to Liu et al. for their original implementation of TLogic.

<h3> How to run </h3>

1. Create a new python environment.
2. Run `poetry install` to install the dependencies from `poetry.lock`.
3. The commands for recreating the results from the paper can be found in `run.txt`.


<h3> Datasets </h3>

Each event in the temporal knowledge graph is written in the format `subject predicate object timestamp`, with tabs as separators.
The dataset is split into `train.txt`, `valid.txt`, and `test.txt`, where we use the same split as provided by [Han et al.](https://github.com/TemporalKGTeam/xERTE)
The files `entity2id.json`, `relation2id.json`, `ts2id.json` define the mapping of entities, relations, and timestamps to their corresponding IDs, respectively.
The file `statistics.yaml` summarizes the statistics of the dataset and is not needed for running the code.


<h3> Parameters </h3>

In `learn.py`:

`--dataset`, `-d`: str. Dataset name.

`--rule_lengths`, `-l`: int. Length(s) of rules that will be learned, e.g., `2`, `1 2 3`.

`--num_walks`, `-n`: int. Number of walks that will be extracted during rule learning.

`--transition_distr`: str. Transition distribution; either `unif` for uniform distribution or `exp` for exponentially weighted distribution.

`--num_processes`, `-p`: int. Number of processes to be run in parallel.

`--seed`, `-s`: int. Random seed for reproducibility.


In `apply.py`:

`--dataset`, `-d`: str. Dataset name.

`--test_data`: str. Data for rule application; either `test` for test set or any other string for validation set.

`--rules`, `-r`: str. Name of the rules file.

`--rule_lengths`, `-l`: int. Length(s) of rules that will be applied, e.g., `2`, `1 2 3`.

`--window`, `-w`: int. Size of the time window before the query timestamp for rule application.

`--top_k`: int. Minimum number of candidates. The rule application stops for a query if this number is reached.

`--num_processes`, `-p`: int. Number of processes to be run in parallel.


In `evaluate.py`:

`--dataset`, `-d`: str. Dataset name.

`--test_data`: str. Data for rule application; either `test` for test set or any other string for validation set.

`--candidates`, `-c`: str. Name of the candidates file.
