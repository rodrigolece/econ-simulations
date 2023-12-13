from matplotlib import pyplot as plt
from typing import Sequence
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import kala

from plotting_functions import plot_dashboard_sbm, plot_dashboard_network
from stats_functions import *


#### SBMs ####
def helper_diagonal_sbm(num_nodes, p_off, p_diag=1.0, seed=0, return_pos=True):
    """Function to create SBM network"""
    n = num_nodes // 2
    sizes = [n, n + num_nodes % 2]

    p_mat = [
        [p_diag, p_off],
        [p_off, p_diag],
    ]

    g = nx.stochastic_block_model(sizes, p_mat, seed=seed)

    if return_pos:
        pos = nx.spring_layout(g, seed=seed)
        out = (g, pos)
    else:
        out = g

    return out


def helper_two_group_assigment(num_nodes, threshold=0.5, seed=0):
    """Function to assign two random blocks of nodes"""
    rng = np.random.default_rng(seed)

    n = num_nodes // 2
    if isinstance(threshold, float):
        block1 = rng.random(size=n) < threshold
        block2 = rng.random(size=n + num_nodes % 2) < 1 - threshold
    elif isinstance(threshold, Sequence) and len(threshold) == 2:
        block1 = rng.random(size=n) < threshold[0]
        block2 = rng.random(size=n + num_nodes % 2) < threshold[1]
    else:
        raise TypeError("treshold must be a float or a sequence of two floats")

    return np.hstack((block1, block2))


def _get_broken_down_wealth_by_savers(game):
    is_saver = [player.get_trait("is_saver") for player in game._get_players()]
    is_saver = np.array(is_saver)
    return (
        game.get_total_wealth(),
        game.get_total_wealth(filt=is_saver),
        game.get_total_wealth(filt=~is_saver),
    )


def _get_broken_down_wealth_by_filter(game, filt):
    return (
        game.get_total_wealth(filt=filt),
        game.get_total_wealth(filt=~filt),
    )


def get_savers_by_cluster(game, filt):
    is_saver = [player.get_trait("is_saver") for player in game._get_players()]
    is_saver = np.array(is_saver)
    savers_1 = sum(is_saver[filt])
    savers_2 = sum(is_saver[~filt])

    return (savers_1, savers_2)


def _get_metrics(game, is_saver=None):
    n = game.get_num_players()
    return (
        game.get_total_wealth() / n,
        game.get_num_savers() / n,
    )


def helper_run_simulation(
    game,
    num_steps,
    cols: Sequence | None = None,
    shocks: Sequence | None = None,
):
    if cols is None:
        cols = ["total", "savers", "non-savers"]

    game.reset_agents()
    data = [_get_broken_down_wealth_by_savers(game)]

    savers = [game.get_num_savers()]

    for step in range(num_steps):
        data.append(_get_broken_down_wealth_by_savers(game))
        savers.append(game.get_num_savers())

        if shocks is not None:
            if shocks[step]:
                for action in shocks[step]:
                    action.apply(game)
        game.play_round()

    return pd.DataFrame(data, columns=cols), savers


def helper_run_simulation_with_filter(
    game,
    num_steps,
    filt,
    cols: Sequence | None = None,
    shocks: Sequence | None = None,
):
    game.reset_agents()

    if len(cols) != 5:
        raise ValueError("length of passed columns argument 'cols' expected to be 5")

    wealth = _get_broken_down_wealth_by_savers(
        game
    ) + _get_broken_down_wealth_by_filter(game, filt)
    data = [wealth]

    savers_by_cluster = get_savers_by_cluster(game, filt)
    tup_savers = tuple([game.get_num_savers()]) + savers_by_cluster
    savers = [tup_savers]

    for step in range(num_steps):
        filt = game.create_filter_from_trait("group", 1)
        filt = np.array(filt, dtype=bool)

        wealth = _get_broken_down_wealth_by_savers(
            game
        ) + _get_broken_down_wealth_by_filter(game, filt)
        data.append(wealth)

        savers_by_cluster = get_savers_by_cluster(game, filt)
        tup_savers = tuple([game.get_num_savers()]) + savers_by_cluster
        savers.append(tup_savers)

        if shocks is not None:
            if shocks[step]:
                for action in shocks[step]:
                    action.apply(game)
        game.play_round()

    return pd.DataFrame(data, columns=cols), pd.DataFrame(
        savers, columns=["total", "cluster_1", "cluster_2"]
    )


def helper_run_simulation_by_metrics(game, num_steps):
    game.reset_agents()

    data = [_get_metrics(game)]

    for _ in range(num_steps):
        game.play_round()
        data.append(_get_metrics(game))

    is_saver_final = [player.get_trait("is_saver") for player in game._get_players()]

    return pd.DataFrame(data, columns=["avg_wealth", "frac_savers"]), is_saver_final


def helper_init(
    g,
    num_players,
    group_thresholds,
    differential_efficient,
    differential_inefficient,
    sigma,
    memory_length,
    update_rule,
    rng=0,
    groups=None,
):
    # Initialise strategy
    strategy = kala.CooperationStrategy(
        stochastic=True,
        rng=rng,
        differential_efficient=differential_efficient,
        differential_inefficient=differential_inefficient,
        dist_sigma_func=sigma,
    )

    # Initialise players
    is_saver = helper_two_group_assigment(num_players, threshold=group_thresholds)
    if groups is None:
        players = [
            kala.InvestorAgent(
                is_saver=s,
                updates_from_n_last_games=memory_length,
                update_rule=update_rule,
            )
            for s in is_saver
        ]
    else:
        players = [
            kala.InvestorAgent(
                is_saver=s,
                group=g,
                updates_from_n_last_games=memory_length,
                update_rule=update_rule,
            )
            for s, g in zip(is_saver, groups)
        ]

    G = kala.SimpleGraph(g, nodes=players)

    # Combine everything and initialise game
    game = kala.DiscreteTwoByTwoGame(G, strategy)

    return game, is_saver  # g, is_saver: returned for plotting


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


def play_game(
    num_players,
    saver_thresh,
    differential_efficient,
    differential_inefficient,
    func,
    memory,
    num_steps,
    num_simulations,
):
    stochastic_strategy = kala.CooperationStrategy(
        stochastic=True,
        rng=0,
        dist_sigma_func=func,
        differential_efficient=differential_efficient,
        differential_inefficient=differential_inefficient,
    )
    g, pos = helper_diagonal_sbm(num_players, p_off=0.1)

    results_mean, results_std = get_mean_std_dfs(
        num_players,
        saver_thresh,
        num_steps,
        num_simulations,
        g,
        stochastic_strategy,
        memory,
    )

    is_saver = helper_two_group_assigment(num_players, threshold=saver_thresh, seed=0)
    plot_network_and_timeseries(results_mean, results_std, is_saver, g, pos)


def get_mean_std_dfs(
    num_players,
    saver_thresh,
    num_steps,
    num_simulations,
    g,
    stochastic_strategy,
    memory=1,
):
    results = pd.DataFrame(index=np.arange(num_steps))
    ### MONTECARLO SIMULATIONS
    for simulation in tqdm(range(num_simulations)):
        is_saver = helper_two_group_assigment(
            num_players, threshold=saver_thresh, seed=simulation
        )

        players = [
            kala.InvestorAgent(is_saver=s, update_from_n_last_games=memory)
            for s in is_saver
        ]
        G = kala.SimpleGraph(g, nodes=players)

        game = kala.DiscreteTwoByTwoGame(G, stochastic_strategy)

        results_dummy = helper_run_simulation(game, is_saver, num_steps=num_steps)
        results = results.join(results_dummy, rsuffix=f"_{simulation}")

    results_mean, results_std = get_mean_and_std_from(results)

    return results_mean, results_std


def plot_network_and_timeseries(means, stds, is_saver, g, pos):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

    ax = axs[0]
    nx.draw(g, node_color=is_saver, pos=pos, ax=ax)
    # ax.set_title(rf"$\tau = {saver_thresh}$")

    ax = axs[1]
    means.plot(ax=ax)
    plot_std(means, stds, alpha=0.3)
    plt.grid()


def get_mean_and_std_from(results, cols=["saver", "non-saver"]):
    results_mean = (
        results[[col for col in results.columns if "total" in col]]
        .mean(axis=1)
        .to_frame("total")
    )
    results_mean[cols[0]] = results[
        [col for col in results.columns if col.startswith("saver")]
    ].mean(axis=1)
    results_mean[cols[1]] = results[
        [col for col in results.columns if "non-saver" in col]
    ].mean(axis=1)

    results_std = (
        results[[col for col in results.columns if "total" in col]]
        .std(axis=1)
        .to_frame("total")
    )
    results_std[cols[0]] = results[
        [col for col in results.columns if col.startswith("saver")]
    ].std(axis=1)
    results_std[cols[1]] = results[
        [col for col in results.columns if "non-saver" in col]
    ].std(axis=1)

    return results_mean, results_std


def plot_std(results_mean, results_std, alpha):
    for col in results_mean.columns:
        plt.fill_between(
            np.arange(len(results_mean)),
            results_mean[col] - results_std[col],
            results_mean[col] + results_std[col],
            alpha=alpha,
        )
