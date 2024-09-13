import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from preprocessing import date_indexer as di

def convert_day_trans_to_week_trans(
        day_trans: pd.DataFrame, date_indexer: di.DateIndexer, 
        prediction_start: str, prediction_end: str,
) -> np.ndarray:
    ''''''
    pnbd_forecasts = []
    for i in range(date_indexer[prediction_start], len(date_indexer) - 1):
        pnbd_forecasts.append(_agg_trans_in_week(day_trans, date_indexer[i], date_indexer[i+1]))
        
    pnbd_forecasts.append(_agg_trans_in_week(
        day_trans, 
        date_indexer[i+1], 
        datetime.strptime(prediction_end, '%Y-%m-%d') + timedelta(days=1))
    )

    return pnbd_forecasts

def _agg_trans_in_week(day_trans: pd.DataFrame, week_start: datetime.timestamp, week_end: datetime.timestamp) -> int:
    ''''''
    total_trans = 0
    curr_day = week_start

    while curr_day < week_end:
        curr_day_row = day_trans[day_trans['period.until'] == curr_day]
        total_trans += curr_day_row['value'].item()

        curr_day += timedelta(days=1)
    
    return total_trans