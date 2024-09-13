import torch
from typing import Callable, List

def calc_RMSE(actual: torch.Tensor, forecast: torch.Tensor) -> float:
    error = 0
    for actual_i, forecast_i in zip(actual, forecast):
        error += (actual_i - forecast_i)**2

    return (error / len(actual))**(1/2)

def calc_MAE(actual: torch.Tensor, forecast: torch.Tensor) -> float:
    error = 0
    for actual_i, forecast_i in zip(actual, forecast):
        error += abs(actual_i - forecast_i)

    return error / len(actual)

def calc_MAPE(actual: torch.Tensor, forecast: torch.Tensor) -> float:
    error = 0
    for actual_i, forecast_i in zip(actual, forecast):
        error += abs((actual_i - forecast_i) / actual_i)

    return 100 * (error / len(actual))

def print_errors(
        actual: torch.Tensor, 
        forecast: torch.Tensor, 
        error_functions: List[Callable[[torch.Tensor, torch.Tensor], float]]=[calc_RMSE, calc_MAE, calc_MAPE]) -> None:
    for func in error_functions:
        error = func(actual, forecast)
        print(f"{func.__name__}: {error}")