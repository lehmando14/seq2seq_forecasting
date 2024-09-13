import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

# check for errors: window_offset, step_size, label_window_size

class TransactionsTimeseriesDataset(Dataset):


    def __init__(
            self, transactions_ts: np.ndarray, max_trans: int,
            input_size: int, label_size: int, 
            label_offset: int, step_size: int=1,            
        ):
        '''https://medium.com/@naveennjn1729/a-quick-introduction-to-time-series-forecasting-b1845beae9b4

        the unit of sizes and offsets are measured in weeks        
        '''
        self.transactions_ts = transactions_ts

        #specifications of the transaction timeseries
        num_customers, num_weeks, num_features = \
            transactions_ts.shape
        self.num_customers = num_customers
        self.num_weeks = num_weeks
        self.num_features = num_features

        #maximum amount of transactions in the dataset
        self.max_trans = max_trans

        #specifications of the sliding window
        self.input_size = input_size
        self.label_size = label_size
        self.label_offset = label_offset
        self.step_size = step_size

        windows_per_customer = \
            (self.num_weeks - (self.input_size + self.label_offset)) // self.step_size + 1
        total_num_windows = windows_per_customer * self.num_customers
        self.windows_per_customer = windows_per_customer
        self.total_num_windows = total_num_windows
        
        
    def __len__(self):
        ''''''
        return self.total_num_windows
    
    def __getitem__(self, i):
        ''''''
        ind = self._global_ind_to_window_ind(i)
        (
            customer_ind,
            start_input,
            end_input,
            start_label,
            end_label,
        ) = ind

        input_ts = self.transactions_ts[customer_ind, start_input:end_input+1]
        #used for backward compatibility
        if self.max_trans > 0:
            label_ts = self.transactions_ts[customer_ind, start_label:end_label+1, :(self.max_trans + 1)]
        else:
            label_ts = self.transactions_ts[customer_ind, start_label:end_label+1]

        return input_ts.astype(np.float32), label_ts.astype(np.float32)
    
    def _global_ind_to_window_ind(self, ind: int) -> Tuple[int, int, int, int, int]:
        '''given an index returns a tuple with 5 integers specifying the window
        
        Returns:
            tuple[0]:   Which customer time series to index
            tuple[1]:   Beginning of the input period (inclusive)
            tuple[2]:   End of the input period (inclusive)
            tuple[3]:   Beginning of the label period (inclusive)
            tuple[4]:   End of the label period (inclusive)
        '''

        customer_ind = ind // self.windows_per_customer

        customer_window_num = ind % self.windows_per_customer
        input_start = self.step_size * customer_window_num
        input_end = input_start + (self.input_size - 1)

        label_end = input_end + self.label_offset
        label_start = label_end - (self.label_size - 1)

        return customer_ind, input_start, input_end, label_start, label_end