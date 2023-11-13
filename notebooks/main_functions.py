import pandas as pd
import numpy as np
from tqdm import tqdm


from helper_functions import (
    helper_diagonal_sbm,
    helper_init,
    helper_run_simulation_with_filter,
    helper_run_simulation,
    helper_run_simulation_by_metrics,
)

from plotting_functions import (
    plot_dashboard_sbm,
    plot_dashboard_network,
    plot_stability_phase_diagrams,
)

from stats_functions import (
    stationary_test_adf,
    stationary_test_std,
    get_time_to_zero,
)

##############################
######### Notebook 1 #########
##############################


def montecarlo_game_sbm(
    num_steps,
    num_simulations,
    num_players,
    thresholds,
    differential_efficient,
    differential_inefficient,
    standard_deviation,
    memory_length,
    update_rule,
):
    g, pos = helper_diagonal_sbm(num_players, p_off=0.1)

    data = [
        num_players,
        num_steps,
        num_simulations,
        differential_efficient,
        differential_inefficient,
        memory_length,
        update_rule.name,
    ]
    rows = [
        "Number of players",
        "Number of steps",
        "Number of simulations",
        "Efficient differential",
        "Inefficient differential",
        "Length of memory",
        "Update rule",
    ]

    table = pd.DataFrame(data, index=rows, columns=["Inputs"])

    wealth_df = pd.DataFrame(index=np.arange(num_steps))
    savers_df = pd.DataFrame(index=np.arange(num_steps))
    for simulation in tqdm(range(num_simulations)):
        game, savers_init = helper_init(
            g,
            num_players,
            thresholds,
            differential_efficient,
            differential_inefficient,
            standard_deviation,
            memory_length,
            update_rule,
        )

        clusters = np.zeros(num_players, dtype=bool)
        clusters[: num_players // 2] = True
        cols_clusters = ["total", "saver", "non-saver", "cluster_1", "cluster_2"]
        df, savers_final = helper_run_simulation_with_filter(
            game, num_steps, clusters, cols_clusters
        )

        wealth_df = wealth_df.join(df, rsuffix=f"_{simulation}")
        savers_df = savers_df.join(savers_final, rsuffix=f"_{simulation}")

    wealth = pd.DataFrame(index=np.arange(num_steps))
    for col_name in cols_clusters:
        mean = (
            wealth_df[[col for col in wealth_df if col_name in col]]
            .mean(axis=1)
            .to_frame(col_name + "_mean")
        )
        std = (
            wealth_df[[col for col in wealth_df if col_name in col]]
            .std(axis=1)
            .to_frame(col_name + "_std")
        )
        wealth = wealth.join(mean)
        wealth = wealth.join(std)

    savers = pd.DataFrame(index=np.arange(num_steps))
    for col_name in ["total", "cluster_1", "cluster_2"]:
        mean = (
            savers_df[[col for col in savers_df if col_name in col]]
            .mean(axis=1)
            .to_frame(col_name + "_mean")
        )
        std = (
            savers_df[[col for col in savers_df if col_name in col]]
            .std(axis=1)
            .to_frame(col_name + "_std")
        )
        savers = savers.join(mean)
        savers = savers.join(std)

    plot_dashboard_sbm(g, table, savers_init, pos, wealth, savers)


def montecarlo_game_network(
    g,
    num_steps,
    num_simulations,
    num_players,
    threshold,
    differential_efficient,
    differential_inefficient,
    standard_deviation,
    memory_length,
    update_rule,
):
    data = [
        num_players,
        num_steps,
        num_simulations,
        differential_efficient,
        differential_inefficient,
        memory_length,
        update_rule.name,
    ]
    rows = [
        "Number of players",
        "Number of steps",
        "Number of simulations",
        "Efficient differential",
        "Inefficient differential",
        "Length of memory",
        "Update rule",
    ]

    table = pd.DataFrame(data, index=rows, columns=["Inputs"])

    wealth_df = pd.DataFrame(index=np.arange(num_steps))
    savers_df = pd.DataFrame(index=np.arange(num_steps + 1))
    for simulation in tqdm(range(num_simulations)):
        game, savers_init = helper_init(
            g,
            num_players,
            threshold,
            differential_efficient,
            differential_inefficient,
            standard_deviation,
            memory_length,
            update_rule,
        )

        df, savers_final = helper_run_simulation(game, num_steps)

        wealth_df = wealth_df.join(df, rsuffix=f"_{simulation}")

        savers_final = pd.DataFrame(
            index=np.arange(num_steps + 1), data=savers_final, columns=["total"]
        )
        savers_df = savers_df.join(savers_final, rsuffix=f"_{simulation}")

    wealth = pd.DataFrame(index=np.arange(num_steps))
    for col_name in ["total", "saver", "non-saver"]:
        mean = (
            wealth_df[[col for col in wealth_df if col_name in col]]
            .mean(axis=1)
            .to_frame(col_name + "_mean")
        )
        std = (
            wealth_df[[col for col in wealth_df if col_name in col]]
            .std(axis=1)
            .to_frame(col_name + "_std")
        )
        wealth = wealth.join(mean)
        wealth = wealth.join(std)

    savers = pd.DataFrame(index=np.arange(num_steps + 1))
    col_name = "savers"
    mean = savers_df.mean(axis=1).to_frame(col_name + "_mean")
    std = savers_df.std(axis=1).to_frame(col_name + "_std")
    savers = savers.join(mean)
    savers = savers.join(std)

    plot_dashboard_network(g, table, savers_init, wealth, savers)


##############################
######### Notebook 3 #########
##############################


def stability_analysis(
    g,
    num_simulations,
    num_steps,
    num_players,
    thresholds,
    standard_deviation,
    memory_length,
    update_rule,
    start,
    end,
    len_linspace,
    timewindow_stability,
    window,
):
    space = np.linspace(start, end, len_linspace)

    final_savers_arr = np.zeros((num_simulations, len_linspace, len_linspace))
    timeseries_savers_arr_adf = np.zeros((num_simulations, len_linspace, len_linspace))
    timeseries_savers_arr_std = np.zeros((num_simulations, len_linspace, len_linspace))
    times_to_zero = np.zeros((num_simulations, len_linspace, len_linspace))

    for sim in tqdm(np.arange(num_simulations)):
        for i, differential_efficient in enumerate(space):
            for j, differential_inefficient in enumerate(space):
                # Init
                game, _ = helper_init(
                    g,
                    num_players,
                    thresholds,
                    differential_efficient,
                    differential_inefficient,
                    standard_deviation,
                    memory_length,
                    update_rule,
                )
                df, savers_final = helper_run_simulation_by_metrics(game, num_steps)

                final_savers_arr[sim, i, j] = df.loc[num_steps, "frac_savers"]
                times_to_zero[sim, i, j] = get_time_to_zero(savers_final)
                timeseries_savers_arr_adf[sim, i, j] = stationary_test_adf(
                    df.loc[:, "frac_savers"].values[-timewindow_stability:]
                )
                timeseries_savers_arr_std[sim, i, j] = stationary_test_std(
                    df.loc[-timewindow_stability:, "frac_savers"], window
                )

    plot_stability_phase_diagrams(
        space,
        final_savers_arr,
        timeseries_savers_arr_adf,
        timeseries_savers_arr_std,
        times_to_zero,
        update_rule.name,
    )
