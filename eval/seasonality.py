import numpy as np

def calculate_seasonality(users_trans_ts: np.ndarray) -> float:
    '''Calculates the seasonality of a time series given the individual non-repeat user
    transactions.

    Parameters:
        user_trans_ts is a three-dimensional array (customer, week, featuers:[transaction_amount, week_number])
    '''

    users_agg_trans_ts = np.sum(users_trans_ts[:, :, 0], axis=0)

    # Calculate the median of the time series
    median = np.median(users_agg_trans_ts)
    
    # Calculate the absolute deviations from the median
    abs_deviations = np.abs(users_agg_trans_ts - median) / median
    
    # Calculate the seasonality as the mean of these absolute deviations
    seasonality = np.mean(abs_deviations)
    
    return seasonality