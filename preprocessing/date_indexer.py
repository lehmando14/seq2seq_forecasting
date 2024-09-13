import pandas as pd

from . import transaction_df_to_timeseries as trans_to_time

class DateIndexer():
    '''The DateIndexer works in the range [earliest_date, latest_date]
    given a date in format yyyy-mm-dd returns the week index containing the date;
    
    given a week index returns the earliest date contained in the week in format yyyy-mm-dd
    '''
    def __init__(self, earliest_date: str, latest_date: str) -> None:
        ''''''

        self.earliest_date = earliest_date

        time_period_frame_df = trans_to_time._create_time_period_frame(earliest_date, latest_date)
        self.time_period_frame_df = time_period_frame_df
        self.agg_time_period_frame_df = self._group_by_week_and_year(time_period_frame_df)

    def __getitem__(self, key):

        if isinstance(key, int):
            return self._ind_to_date(key)
        if isinstance(key, str):
            return self._date_to_ind(key)        
        raise TypeError("Input is neither an int nor a str")
    
    def __len__(self):
        return len(self.agg_time_period_frame_df)
    

    def _group_by_week_and_year(self, time_period_frame_df: pd.DataFrame):
        agg_time_period_frame_df = time_period_frame_df.groupby(['year', 'week']).agg({
            'date': 'min'
        }).reset_index()

        return agg_time_period_frame_df
    
    def _date_to_ind(self, date: str):
        date_ind = self.time_period_frame_df['date'] == date
        year = self.time_period_frame_df[date_ind]['year'].values[0]
        week = self.time_period_frame_df[date_ind]['week'].values[0]

        year_ind = self.agg_time_period_frame_df['year'] == year
        week_ind = self.agg_time_period_frame_df['week'] == week
                
        return self.agg_time_period_frame_df[year_ind & week_ind].index[0]

    def _ind_to_date(self, ind: int):
        return self.agg_time_period_frame_df.iloc[ind]['date']