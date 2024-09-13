import pandas as pd
import numpy as np
from datetime import datetime

from preprocessing import transaction_df_to_timeseries as transaction_df_to_timeseries
from preprocessing import date_indexer as di
import displays

def test_date_indexer():
    data = {
    'account_id': [101, 101, 101, 101, 102, 102],
    'date': [
        '2024-01-05', '2024-01-12', '2024-01-29', '2024-02-02',
        '2024-01-03', '2024-01-28'
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    users_trans = transaction_df_to_timeseries.transform_trans_df_to_time_series(df, start='2024-01-01', first_purch_before='2024-02-28', end='2024-02-28')

    date_indexer = di.DateIndexer('2024-01-01', '2024-02-28')

    #date to ind
    assert date_indexer['2024-01-27'] == 3
    assert date_indexer['2024-01-02'] == 0
    assert date_indexer['2024-01-14'] == 2

    #ind to date
    assert date_indexer[1] == datetime(2024, 1, 7)
    assert date_indexer[4] == datetime(2024, 1, 28)