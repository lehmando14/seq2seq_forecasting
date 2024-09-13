import numpy as np
import torch

def calculate_losses(
        pred_trans: torch.Tensor, actual_trans: torch.Tensor, loss_funcs: list,
        date_indexer, start_test_period: str, end_test_period: str
    ) -> list:
    '''calculates losses with specifying a loss period, pred_trans and actual_trans
    both have the shape (customers, weeks)
    '''

    ind_start_test = date_indexer[start_test_period]
    ind_end_test = date_indexer[end_test_period]

    pred_trans_test_period = pred_trans[:, ind_start_test:ind_end_test+1]
    actual_trans_test_period = actual_trans[:, ind_start_test:ind_end_test+1]

    return calculate_losses_whole(pred_trans_test_period, actual_trans_test_period, loss_funcs)

def calculate_losses_whole(pred_trans: torch.Tensor, actual_trans: torch.Tensor, loss_funcs: list) -> list:
    '''calculates losses without specifying a test period'''
    agg_pred_trans = create_agg_weekly_trans(pred_trans)
    agg_actual_trans = create_agg_weekly_trans(actual_trans)
    losses = []
    for loss_func in loss_funcs:
        loss = _calc_loss_agg_weekly_trans(agg_pred_trans, agg_actual_trans, loss_func)
        losses.append(loss)

    return losses

def create_agg_weekly_trans(users_ts: torch.Tensor) -> torch.Tensor:
    '''Aggregates all the transactions of each customer per week
    Input:
        users_ts: amount of transactions of a customer in a certain week; dim (customer, week)

    Return:
        amount of transactions in each week; dim (week)
    '''
    return torch.sum(users_ts, dim=0)


def _calc_loss_agg_weekly_trans(pred_trans: torch.Tensor, actual_trans: torch.Tensor, loss_func):
    ''''''
    return loss_func(pred_trans, actual_trans)