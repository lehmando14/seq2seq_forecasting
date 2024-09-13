import numpy as np
from tqdm.auto import tqdm

def time_series_to_one_hot(ts: np.ndarray, max_trans: int) -> np.ndarray:
    '''replaces amount of transactions per week with one hot encoding'''
    num_customers, num_weeks, _ = ts.shape
    new_ts = np.zeros([num_customers, num_weeks, (max_trans + 1) + 52], dtype=np.ubyte)
        
    for customer_ind in tqdm(range(num_customers), desc='Converting to one-hot'):
        for week_ind in range(num_weeks):
            #create one-hot for amount of transactions
            one_hot_trans_ind = ts[customer_ind, week_ind, 0]
            new_ts[customer_ind, week_ind, one_hot_trans_ind] = 1

            #create one-hot for week
            one_hot_week_ind = ts[customer_ind, week_ind, 1] + (max_trans + 1)
            new_ts[customer_ind, week_ind, one_hot_week_ind] = 1

    return new_ts