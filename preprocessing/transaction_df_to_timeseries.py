import pandas as pd
import numpy as np
import datetime
from tqdm.auto import tqdm


def transform_trans_df_to_time_series(
        trans_df: pd.DataFrame, start: str, first_purch_before: str, end: str
    ) -> np.ndarray:
    '''returns the time series of all customer transactions between [start, end], where
    the first purchase happened before and including "first_purch_before"

    Parameters:
        trans_df:   (add later)
        start:      example "1995-12-31"
        end:        example "1996-12-31"

    Returns:        (user_trans_ts) ndarray with shape (x, y, z).
                        x: amount of different customers
                        y: amount of weeks in a timeseries
                        z: amount of features - example 2 for transaction amount and week

                    (ids) ndarray with shape (x).
                        ids[i] returns the customer id of the customer at
                        user_trans_ts[i]
                           
    '''
    users_trans_ts = []

    trans_df = trans_df.copy(deep=True)
    trans_df = _get_customers_before_date(trans_df, first_purch_before)
    ids = _get_account_ids(trans_df)

    time_period_frame_df = _create_time_period_frame(start, end)

    for account in tqdm(ids, desc='Preparing dataset'):
        user_trans_df = _get_user_transactions_df(trans_df, account)
        user_weekly_trans_df = _calculate_user_weekly_transactions_df(
            user_trans_df, time_period_frame_df
        )
        user_weekly_trans_ts = _create_time_series(user_weekly_trans_df)
        users_trans_ts.append(user_weekly_trans_ts)

    users_trans_ts = np.stack(users_trans_ts)

    return users_trans_ts, ids

def transform_trans_df_to_time_series_repeat(
        trans_df: pd.DataFrame, start: str, first_purch_before: str, end: str
    ) -> np.ndarray:
    '''returns the time series of all customer transactions between [start, end], where
    the first purchase happened before and including "first_purch_before". The 
    difference between transform_trans_df_to_time_series_repeat and
    transform_trans_df_to_time_series is that the former only considers
    repeat purchases.

    Parameters:
        trans_df:   (add later)
        start:      example "1995-12-31"
        end:        example "1996-12-31"

    Returns:        (user_trans_ts) ndarray with shape (x, y, z).
                        x: amount of different customers
                        y: amount of weeks in a timeseries
                        z: amount of features - example 2 for transaction amount and week

                    (ids) ndarray with shape (x).
                        ids[i] returns the customer id of the customer at
                        user_trans_ts[i]
                           
    '''
    users_trans_ts = []

    trans_df = trans_df.copy(deep=True)
    trans_df = _get_customers_before_date(trans_df, first_purch_before)
    ids = _get_account_ids(trans_df)

    time_period_frame_df = _create_time_period_frame(start, end)

    for account in tqdm(ids, desc='Preparing dataset'):
        user_trans_df = _get_user_transactions_df(trans_df, account)
        user_trans_df = _remove_first_transaction(user_trans_df)
        user_weekly_trans_df = _calculate_user_weekly_transactions_df(
            user_trans_df, time_period_frame_df
        )
        user_weekly_trans_ts = _create_time_series(user_weekly_trans_df)
        users_trans_ts.append(user_weekly_trans_ts)

    users_trans_ts = np.stack(users_trans_ts)

    return users_trans_ts, ids


def _get_customers_before_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    '''returns all customers who made their first pruchase before and including date, i.e. they
    have been active before date
    '''
    df = df.copy(deep=True)
    cohort_accounts = df.groupby('account_id').min().query(
    'date <= @date').reset_index()['account_id'].tolist()

    df = df.query('account_id in @cohort_accounts')
    df = df.sort_values(by='account_id').reset_index(drop=True)

    print(f"Accounts in dataset:  {len(df['account_id'].unique())}")
    print(f"Total transactions: {len(df)}")

    return df

def _get_account_ids(df: pd.DataFrame) -> list:
    '''returns a list of unique account ids from a transaction DataFrame'''
    return df['account_id'].unique()

def _create_time_period_frame(start: str, end: str) -> pd.DataFrame:
    '''creates a dataframe with entries corresponding to all dates between [start and end]'''
    date_range = _create_date_range(start, end)
    date_range_template_df = _create_template_df(date_range)

    return date_range_template_df

def _get_user_transactions_df(df: pd.DataFrame, account_id) -> pd.DataFrame:
    '''given a transaction df of all users creates a df with the amount of 
    transactions per date for a specific user
    '''
    subset = df.query('account_id == @account_id').groupby(
        ['date']).count().reset_index()
    user_transactions_df = subset.copy(deep=True)
    user_transactions_df = user_transactions_df.rename(columns={'account_id': 'transactions'})

    return user_transactions_df

def _calculate_user_weekly_transactions_df(
        user_transactions_df: pd.DataFrame, date_template_df: pd.DataFrame
    ) -> pd.DataFrame:
    '''Returns weekly transactions of a particalur customer in chronological order 
    in a given time period

    Parameters:
        user_transaction_df - All transactions a user has made
        date_template_df - Dataframe containing (year, week) pairs within a given timespan.
            Is used to select all user transactions happening within given timespan.
    '''
    frame = date_template_df.copy(deep=True)

    user_weekly_trans_df = frame.merge(user_transactions_df, on=['date'], how='left')
    user_weekly_trans_df = user_weekly_trans_df.groupby(['year', 'week']).agg(
        {'transactions': 'sum', 'date': 'min'}
    )
    user_weekly_trans_df = user_weekly_trans_df.sort_values(['date']).reset_index()
    return user_weekly_trans_df

def _create_time_series(user_weekly_trans_df: pd.DataFrame) -> list[list]:
    '''Transforms weekly transaction into a time series: 
    [(transactions_0, ..., week_0), ...., (transactions_n, ..., week_n)]
    '''
    user_weekly_trans_df = user_weekly_trans_df[['transactions', 'week']].astype(int)
    return user_weekly_trans_df.values


def _create_date_range(start: str, end: str, date_format='%Y-%m-%d') -> list[datetime.datetime]:
    '''generate a list of dates between two dates'''
    start = datetime.datetime.strptime(start, date_format)
    end = datetime.datetime.strptime(end, date_format)
    r = (end+datetime.timedelta(days=1)-start).days
    return [start+datetime.timedelta(days=i) for i in range(r)]

def _create_template_df(date_range: list[datetime.datetime]) -> pd.DataFrame:
    '''takes list of date between two dates and creates a dataframe with additional columns: week & year'''
    template_df = pd.DataFrame(date_range, columns=['date'])
    template_df['year'] = template_df['date'].dt.year      # 0-indexing 
    template_df['week'] = (template_df['date'].dt.dayofyear // 7).clip(upper=51)

    return template_df

def _remove_first_transaction(user_train_df: pd.DataFrame) -> pd.DataFrame:
    '''Removes the first transaction for the customer in the provided DataFrame, 
    ensuring the transactions are sorted by date.
    '''
    # Ensure the DataFrame is sorted by date
    user_train_df = user_train_df.sort_values(by='date').reset_index(drop=True)
    
    # Drop the first row
    user_train_df = user_train_df.iloc[1:].reset_index(drop=True)
    
    return user_train_df