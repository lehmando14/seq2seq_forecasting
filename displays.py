import pandas as pd
import matplotlib.pyplot as plt

def count_aggregate_weekly_transactions(df: pd.DataFrame) -> pd.DataFrame:
    '''returns dataframe with amount of weekly transactions from dataset'''
    aggregate_counts = df.copy(deep=True)
    aggregate_counts['year'] = aggregate_counts['date'].dt.year
    aggregate_counts['week'] = (aggregate_counts['date'].dt.dayofyear // 7).clip(upper=51) # we roll the 52nd week into the 51st
    aggregate_counts = aggregate_counts.groupby(['year', 'week']).agg({'account_id': 'count', 'date': 'min'}).reset_index()
    aggreate_weekly_transactions = aggregate_counts.rename(columns={"account_id": "weekly_transactions"})

    return aggreate_weekly_transactions

def display_aggregate_weekly_transactions(df: pd.DataFrame, training_end: str, holdout_start: str) -> pd.DataFrame:
    '''displays a plot of the weekly aggregate transactions. The plot 
    distinguishes between training and test period
    '''
    plt.figure(figsize=(16,5))
    plt.plot(df[df.date <= training_end].index, 
         df[df.date <= training_end]['weekly_transactions'], color='black', label='calibration')
    plt.plot(df[df.date >= holdout_start].index, 
         df[df.date >= holdout_start]['weekly_transactions'], color='blue', label='holdout')
    plt.axvline(len(df[df.date <= training_end]), linestyle=':')
    plt.title('Weekly Aggregate Transactions - Calibration and Holdout')
    plt.legend()
    plt.savefig("calibration_holdout.png", dpi=600)
    plt.show()

def display_total_trans_dist(df: pd.DataFrame) -> pd.DataFrame:
    '''Creates a histogram of the amount of transactions per customer'''
    df = df.copy(deep=True)
    transactions_per_account = list(df.groupby('account_id').count()['date'])
    plt.figure(figsize=(16,5))
    plt.hist(transactions_per_account, bins=250, color='black')
    plt.title('Total Transaction Counts per Account - Frequency Histogram')
    plt.savefig("transactions_histogram.png", dpi=600)
    plt.show()