# Shocks

This document is a detailed list of the shocks that have been implemented until now.

### Importation

For an easy importation of all the shocks available, please add this line to any notebook or python script:
```python
from kala.models.shocks import *
```

If only a subset of shocks is needed, the above line can be changed to:
```python
from kala.models.shocks import Shock1, Shock2, ...
```

### General Structure

Every shock has a similar structure that can be implemented in the following way:
```python
Shock(args).apply(game)
```
In this case, we refer to a generic function [`Shock()`](https://github.com/rodrigolece/kala-econ-games/blob/main/src/kala/models/shocks.py) with arguments `args` that is applied, via the method `apply()`, to the [`game`](https://github.com/rodrigolece/kala-econ-games/blob/main/src/kala/models/game.py) we are studying.

Shocks can be applied at any given point in time during a simulation. Also, multiple shocks can be sequentially applied in a particular time step. To achieve that, we just need to create a list with length equal to the duration of the simulation (`num_steps`). This can be achieved in the following way:

```python
shocks = [[] for t in range(num_steps)]

# Add a change of memory length to all players at time t=5
shocks[5].append(ChangeAllPlayersMemoryLength(new_memory_length=10))

# Add a sequence of two shocks at time t=10
shocks[10].append(RemoveRandomEdge())
shocks[10].append(FlipAllSavers())
```

### List of Shocks

#### Inputs
The list below makes use of different inputs. Here are the kind of variables they are:
- `player`: an integer, a string or a node (`AgentT`).
- `list_of_players`: a list, tuple or `np.array` of players.
- `saver_trait`: a boolean (`True` or `False`). `True` designates a `saver`, while `False` designates a `non-saver`.
- `new_memory_length`: a positive integer refering the number of rounds a player remembers.
- `new_parameter`: a float number refering to the [`differential` parameters](https://github.com/rodrigolece/kala-econ-games/blob/main/src/kala/models/strategies.py)

#### Shocks to the structure of the network
1. `RemovePlayer(player)`. Remove a given node `player` from the network. The shock also removes the adjacent edges to the selected node.
    :warning: RemovePlayer has a bug that does not allow to analyse the network after the shock. A shock so strong that even we do not know what to do.
2. `RemoveRandomPlayer()`. Remove a random node from the nework. The shock also removes the adjacent edges to the random node.
    :warning: RemovePlayer has a bug that does not allow to analyse the network after the shock. A shock so strong that even we do not know what to do.
3. `RemoveEdge(player1, player2)`. Remove the edge between `player1` and `player2` from the network.
4. `RemoveRandomEdge()`. Removes a random edge from the network.
5. `SwapEdge(pivot, player1, player2)`. Remove edge `(pivot, player1)` and add edge `(pivot, player2)` to the network.
6. `SwapRandomEdge()`. Swap an edge between three random players as for the shock `SwapEdge()`.
7. `AddEdge(player1, player2)`. Add an edge between `player1` and `player2` to the network.
8. `AddRandomEdge()`. Add an edge between two random players in the network.

#### Shocks to the saver traits of players
9. `FlipSaver(player)`. Flip a player's saver trait. If `player` is a `saver`, then the shock would make it a `non-saver`, and vice-versa.
10. `FlipRandomSaver()`. Flip a random player's saver trait. If the selected random `player` is a `saver`, then the shock would make it a `non-saver`, and vice-versa.
11. `FlipSavers(list_of_players)`. Flip the saver traits of a given list of players in the network.
12. `FlipAllSavers()`. Flip the saver traits of all players in the network.
13. `HomogenizeSaversTo(saver_trait)`. Change all players' saver traits to a given `saver_trait` that takes as values: `True` for changing all players to `saver` or `False` to change all players to `non-saver`.

#### Shocks to a player's memory length
14. `ChangePlayerMemoryLength(player, new_memory_length)`. Change a `player`'s memory length to `new_memory_length`.
15. `ChangeRandomPlayerMemoryLength(new_memory_length)`. Change a random player's memory length to `new_memory_length`.
16. `ChangeAllPlayersMemoryLength(new_memory_length)`. Change all players' memory length to `new_memory_length`.

#### Shocks to the differential parameters (eta_hat and eta_hat_hat)
17. `ChangeDifferentialEfficient(new_parameter)`: Change the `differential_efficient` (eta_hat_hat) parameter of the game to `new_parameter`.
18. `ChangeDifferentialInefficient(new_parameter)`: Change the `differential_inefficient` (eta_hat) parameter of the game to `new_parameter`.

 