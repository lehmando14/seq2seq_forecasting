import torch

from eval import error_functions as ef

def test_error_functions():

    actual = torch.tensor([1.0, 3.0])
    forecast = torch.tensor([1.0, 6.0])

    assert abs(ef.calc_RMSE(actual, forecast) - 2.1213) < 0.001
    assert abs(ef.calc_MAE(actual, forecast) - 1.5) < 0.001
    assert abs(ef.calc_MAPE(actual, forecast) - 50) < 0.001