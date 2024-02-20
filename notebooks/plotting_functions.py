import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import numpy as np
import pandas as pd


def plot_mean_and_std(df, label, color, ax):
    col = label.lower().replace(" ", "_")
    ax.plot(df[col + "_mean"], label=label, color=color)
    num_steps = len(df[col + "_mean"])
    ax.fill_between(
        np.arange(num_steps),
        df[col + "_mean"] - df[col + "_std"],
        df[col + "_mean"] + df[col + "_std"],
        color=color,
        alpha=0.5,
    )


def plot_dashboard_sbm(g, table, savers_init, pos, wealth, savers):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))

    ax[0, 0].axis("off")
    ax[0, 0].axis("tight")
    ax[0, 0].table(
        cellText=table.values,
        rowLabels=table.index,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        # colWidths=[0.4, 0.2],
        fontsize=15.0,
    )

    nx.draw(g, node_color=savers_init, ax=ax[0, 1], pos=pos)
    ax[0, 1].set_title("Initial state network")

    plot_mean_and_std(wealth, "Total", "b", ax[1, 0])
    plot_mean_and_std(wealth, "Saver", "g", ax[1, 0])
    plot_mean_and_std(wealth, "Non-Saver", "r", ax[1, 0])
    ax[1, 0].set_xlabel("Step")
    ax[1, 0].set_ylabel("Wealth")
    ax[1, 0].grid(alpha=0.4)
    ax[1, 0].set_title("Wealth of savers and non-savers")
    ax[1, 0].legend()

    plot_mean_and_std(wealth, "Total", "b", ax[1, 1])
    plot_mean_and_std(wealth, "Cluster 1", "orange", ax[1, 1])
    plot_mean_and_std(wealth, "Cluster 2", "m", ax[1, 1])
    ax[1, 1].set_xlabel("Step")
    ax[1, 1].set_ylabel("Wealth")
    ax[1, 1].set_title("Wealth of clusters")
    ax[1, 1].grid(alpha=0.4)
    ax[1, 1].legend()

    plot_mean_and_std(savers, "Total", "b", ax[2, 0])
    plot_mean_and_std(savers, "Cluster 1", "orange", ax[2, 0])
    plot_mean_and_std(savers, "Cluster 2", "m", ax[2, 0])
    ax[2, 0].grid(alpha=0.4)
    ax[2, 0].set_ylabel("Number of savers in the game")
    ax[2, 0].set_xlabel("Step")
    ax[2, 0].set_title("evolution of savers")
    ax[2, 0].legend()

    ax[2, 1].axis("off")


def plot_dashboard_network(g, table, savers_init, wealth, savers):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 16))

    ax[0, 0].axis("off")
    ax[0, 0].axis("tight")
    ax[0, 0].table(
        cellText=table.values,
        rowLabels=table.index,
        colLabels=table.columns,
        loc="center",
        cellLoc="left",
        # colWidths=[0.4, 0.2],
    )

    nx.draw(g, node_color=savers_init, ax=ax[0, 1])
    ax[0, 1].set_title("Initial state network")

    plot_mean_and_std(wealth, "Total", "b", ax[1, 0])
    plot_mean_and_std(wealth, "Saver", "g", ax[1, 0])
    plot_mean_and_std(wealth, "Non-Saver", "r", ax[1, 0])
    ax[1, 0].set_xlabel("Step")
    ax[1, 0].set_ylabel("Wealth")
    ax[1, 0].grid(alpha=0.4)
    ax[1, 0].set_title("Wealth of savers and non-savers")
    ax[1, 0].legend()

    plot_mean_and_std(savers, "Savers", "b", ax[1, 1])
    ax[1, 1].grid(alpha=0.4)
    ax[1, 1].set_ylabel("Number of savers in the game")
    ax[1, 1].set_xlabel("Step")
    ax[1, 1].set_title("evolution of savers")
    ax[1, 1].legend()


def plot_stability_phase_diagrams(
    space,
    final_savers_arr,
    timeseries_savers_arr_adf,
    timeseries_savers_arr_std,
    times_to_zero,
    update_rule_name,
):
    final = final_savers_arr.mean(axis=0)
    stability_arr_adf = timeseries_savers_arr_adf.mean(axis=0)
    stability_arr_std = timeseries_savers_arr_std.mean(axis=0)
    times_to_zero_mean = times_to_zero.mean(axis=0)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ticks = np.round(space, 2)

    num_steps = len(space)
    sns.heatmap(
        times_to_zero_mean,
        vmin=-num_steps / 3,
        vmax=num_steps,
        ax=ax[0, 0],
        xticklabels=ticks,
        yticklabels=ticks,
    )
    ax[0, 0].set_title("Average time to zero")
    ax[0, 0].set_xlabel("differential_efficient")
    ax[0, 0].set_ylabel("differential_inefficient")

    sns.heatmap(
        final, vmin=0, vmax=1.0, ax=ax[0, 1], xticklabels=ticks, yticklabels=ticks
    )
    ax[0, 1].set_title(update_rule_name)
    ax[0, 1].set_xlabel("differential_efficient")
    ax[0, 1].set_ylabel("differential_inefficient")

    sns.heatmap(
        stability_arr_adf,
        vmin=0,
        vmax=1.0,
        ax=ax[1, 0],
        xticklabels=ticks,
        yticklabels=ticks,
    )
    ax[1, 0].set_title("Stability")
    ax[1, 0].set_xlabel("differential_efficient")
    ax[1, 0].set_ylabel("differential_inefficient")

    sns.heatmap(
        stability_arr_std,
        vmin=0,
        vmax=1.0,
        ax=ax[1, 1],
        xticklabels=ticks,
        yticklabels=ticks,
    )
    ax[1, 1].set_title("S.D. Stability")
    ax[1, 1].set_xlabel("differential_efficient")
    ax[1, 1].set_ylabel("differential_inefficient")
