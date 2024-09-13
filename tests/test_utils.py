import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing import transaction_df_to_timeseries as transaction_df_to_timeseries
from preprocessing import date_indexer as di
import displays

def test_get_trainingset():
    df = pd.read_csv(filepath_or_buffer='_data/trans.csv', 
                 usecols=['account_id', 'date'], 
                 parse_dates=['date'])
    
    training_end   = '1995-12-31'
    training_df = transaction_df_to_timeseries._get_customers_before_date(df, training_end)

    assert len(training_df['account_id'].unique()) == 2239
    assert len(training_df) == 744015
    
def test_count_aggregate_weekly_transactions():

    df = pd.read_csv(filepath_or_buffer='_data/trans.csv', 
                 usecols=['account_id', 'date'], 
                 parse_dates=['date'])
    
    training_end   = '1995-12-31'
    training_df = transaction_df_to_timeseries._get_customers_before_date(df, training_end)


    aggregate_counts = displays.count_aggregate_weekly_transactions(training_df)
    aggregate_counts.to_csv('tests/data/aggregate_counts.csv', index=False)

    anchor_aggregate_counts = pd.read_csv('tests/data/aggregate_counts.csv', parse_dates=['date'])
    assert aggregate_counts.equals(anchor_aggregate_counts)

def test_calculate_user_weekly_transactions_df():

    df = pd.read_csv(filepath_or_buffer='_data/trans.csv', 
                 usecols=['account_id', 'date'], 
                 parse_dates=['date'])

    training_start = '1993-01-01'
    training_end   = '1995-12-31'
    holdout_end    = '1998-12-31'

    date_range = transaction_df_to_timeseries._create_date_range(training_start, holdout_end)
    date_range_template_df = transaction_df_to_timeseries._create_template_df(date_range)

    training_df = transaction_df_to_timeseries._get_customers_before_date(df, training_end)

    user_transactions_df = transaction_df_to_timeseries._get_user_transactions_df(training_df, 234)
    user_weekly_trans_df = transaction_df_to_timeseries._calculate_user_weekly_transactions_df(
        user_transactions_df, date_range_template_df
    )
    user_weekly_trans_df = user_weekly_trans_df.drop(columns=['transactions'])

    user_transactions_df2 = transaction_df_to_timeseries._get_user_transactions_df(training_df, 237)
    user_weekly_trans_df2 = transaction_df_to_timeseries._calculate_user_weekly_transactions_df(
        user_transactions_df2, date_range_template_df
    )
    user_weekly_trans_df2 = user_weekly_trans_df2.drop(columns=['transactions'])

    assert user_weekly_trans_df.equals(user_weekly_trans_df2)

def test_transform_trans_df_to_time_series():
    '''tests if transformation form transactions df to users time series works'''

    #user 101
    array1 = np.array([[1, 0],
                    [1, 1],
                    [0, 2],
                    [0, 3],
                    [2, 4],
                    [0, 5],
                    [0, 6],
                    [0, 7],
                    [0, 8]])

    #user 102
    array2 = np.array([[1, 0],
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [0, 5],
                    [0, 6],
                    [0, 7],
                    [0, 8]])

    array_list = np.stack((array1, array2))

    data = {
    'account_id': [101, 101, 101, 101, 102, 102, 103],
    'date': [
        '2024-01-05', '2024-01-12', '2024-01-29', '2024-02-02',
        '2024-01-02', '2024-01-28',
        '2024-02-18',
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    users_trans, ids = transaction_df_to_timeseries.transform_trans_df_to_time_series(df, start='2024-01-01', first_purch_before='2024-02-17', end='2024-02-28')

    assert ids[0] == 101
    assert ids[1] == 102
    assert np.array_equal(array_list, users_trans)