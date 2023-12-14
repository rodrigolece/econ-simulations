import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def stationary_test_adf(observations):
    """
    function to test stationarity using an
    augmented Dickey-Fuller test with a 1% significance
    """
    stationarity = True

    if observations.min() != observations.max():
        results = adfuller(observations)

        adf = results[0]
        p_value = results[1]
        critical_value_001 = results[4]["1%"]

        if (adf < critical_value_001) and (p_value < 0.01):
            return stationarity
        else:
            return not stationarity
    else:
        return stationarity


def stationary_test_std(observations: pd.DataFrame, window):
    """
    function to test stationarity of the standard deviation using an
    augmented Dickey-Fuller test with a 1% significance
    """
    rolled_std = observations.rolling(window).std().dropna()
    stationarity = stationary_test_adf(rolled_std)

    return stationarity


def get_time_to_zero(observations):
    times = np.where(observations == 0)[0]
    if len(times) != 0:
        min_time = min(times)
    else:
        min_time = -1
    return min_time
