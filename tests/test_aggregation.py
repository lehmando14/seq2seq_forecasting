import pandas as pd
import numpy as np
import torch

from preprocessing import transaction_df_to_timeseries as transaction_df_to_timeseries
from preprocessing import date_indexer as di
from clv_prediction import aggregation

def test_calculate_losses():

    actual_trans = torch.tensor([
        [0, 0, 1],
        [0, 2, 1],
        [1, 3, 0],
    ]).to(torch.float32)

    pred_trans = torch.tensor([
        [0, 0, 1],
        [0, 2, 1],
        [1, 0, 0],
    ]).to(torch.float32)

    loss_funcs = [torch.nn.L1Loss(), torch.nn.MSELoss()]

    loss1, loss2 = aggregation.calculate_losses_whole(pred_trans, actual_trans, loss_funcs)
    assert loss1 == torch.Tensor([1.0])
    assert loss2 == torch.Tensor([3.0])

def test_whole():
    _test_1()
    _test_2()
    _test_3()


def _test_1():
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
    users_trans, users_ids = transaction_df_to_timeseries.transform_trans_df_to_time_series(
        df, start='2024-01-01', first_purch_before='2024-02-28', end='2024-02-28'
    )
    actual_trans = torch.Tensor(users_trans[:, :, 0])

    pred_trans = torch.clone(actual_trans)
    pred_trans[0, 2] = 1

    date_indexer = di.DateIndexer('2024-01-01', '2024-02-28')

    losses = aggregation.calculate_losses(
        pred_trans, actual_trans, [torch.nn.L1Loss(), torch.nn.MSELoss()], date_indexer,
        start_test_period='2024-01-14', end_test_period='2024-01-14'    
    )

    assert losses[0] == torch.Tensor([1.0])
    assert losses[1] == torch.Tensor([1.0])

def _test_2():
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
    users_trans, users_ids = transaction_df_to_timeseries.transform_trans_df_to_time_series(
        df, start='2024-01-01', first_purch_before='2024-02-28', end='2024-02-28'
    )
    actual_trans = torch.Tensor(users_trans[:, :, 0])

    pred_trans = torch.clone(actual_trans)
    pred_trans[:, 2] = 1
    pred_trans[0, 4] = 3

    date_indexer = di.DateIndexer('2024-01-01', '2024-02-28')

    losses = aggregation.calculate_losses(
        pred_trans, actual_trans, [torch.nn.L1Loss(), torch.nn.MSELoss()], date_indexer,
        start_test_period='2024-01-14', end_test_period='2024-01-27'    
    )

    assert losses[0] == torch.Tensor([1.0])
    assert losses[1] == torch.Tensor([2.0])

def _test_3():
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
    users_trans, users_ids = transaction_df_to_timeseries.transform_trans_df_to_time_series(
        df, start='2024-01-01', first_purch_before='2024-02-28', end='2024-02-28'
    )
    actual_trans = torch.Tensor(users_trans[:, :, 0])

    pred_trans = torch.clone(actual_trans)
    pred_trans[:, 2] = 1
    pred_trans[0, 4] = 3

    date_indexer = di.DateIndexer('2024-01-01', '2024-02-28')

    losses = aggregation.calculate_losses(
        pred_trans, actual_trans, [torch.nn.L1Loss(), torch.nn.MSELoss()], date_indexer,
        start_test_period='2024-01-28', end_test_period='2024-02-05'    
    )

    assert losses[0] == torch.Tensor([0.5])
    assert losses[1] == torch.Tensor([0.5])