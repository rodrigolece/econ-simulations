# Networks

This document is a minimal list of networks that can be implemented from `networkx` to the simulations present in this repository. The networks present here are just a small subset of networks allowed by `networkx`.

For a complete list, please refer to the `networkx` [documentation](https://networkx.org/documentation/stable/reference/generators.html).

### Importation

To import `networkx` to any notebook or python script, please include the next line:
```python
import networkx as nx
```

### List of networks

Every network listed below, with the exception of the stochastic block model, need to be fed to the function `montecarlo_game_network()` as a first argument `g`. e.g.:
```python
g = nx.barabasi_albert_graph(num_players, num_edges)

montecarlo_game_network(g, ...)
```


1. Stochastic Bloc Model (SBM)

We have done our own implementation of the SBM graph. The main function `montecarlo_game_sbm()` initiates an SBM graph with two clusters of equal size `num_players//2`. Each cluster is complete, meaning that their edge densities is 1. Between clusters, the density of edges is 0.1

2. Erdös-Renyi random graph

To create an Erdös-Renyi random graph with _n_ nodes and probability of edge creation _p_, we need to write:
```python
num_playes = n # number of nodes
p_edges = p # probability of edge creation

g = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
```

The two extra terms, `seed` and `directed`, respectively refer to the seed of the random number generator and to a boolean reflecting if the output network is directed or not.

There exists other similiar versions of this network in the `networkx` [documentation](https://networkx.org/documentation/stable/reference/generators.html).

3. Barabasi-Albert graph

To create a Barabasi-Albert random graph with _n_ nodes and _e_ edges, we need to write:
```python
num_players = n
num_edges = e

g = nx.barabasi_albert_graph(num_players, num_edges, seed=seed)
```

`seed` refers to the seed of the random number generator.

There exists other similiar versions of this network in the `networkx` [documentation](https://networkx.org/documentation/stable/reference/generators.html).

4. Connected Small-World graph

To create a small-world graph with _n_ nodes with _k_ neighbours and probability _p_ of rewiring, we write:
```python
num_players = n
num_neighbours = k
prob_rewiring = p

g = nx.connected_watts_strogratz_graph(num_players, num_neighbours, prob_rewiring, tries=100, seed=seed)
```

`seed` refers to the seed of the random number generator. `tries` referes to the maximum number of tries that the algorithm tries to create the network. The default value of this argument is 100.

There exists other similiar versions of this network in the `networkx` [documentation](https://networkx.org/documentation/stable/reference/generators.html).

5. Regular random graph

To create a regular random graph with degree _k_ and _n_ nodes, we need to write:
```python
degree = k
num_players = n

g = nx.random_regular_graph(degree, num_players, seed=seed)
```
`seed` refers to the seed of the random number generator.
