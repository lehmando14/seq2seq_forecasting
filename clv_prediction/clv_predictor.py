import numpy as np
import torch
from typing import Tuple
import torch.nn.functional as F

class CLVPredictor():
    '''Facade class to create predictions

    Attributes:
        pred_model (GRU): ML model used to predict future transactions
        user_ts (tensor): 3-dim tensor (customers, weeks, one-hot of week & transactions); this tensor should contain
            all the transactions available
        users_ts_ids (ndarray): numpy array mapping customer ind to customer id
        date_ind (DateIndexer): translates date to index in (_, weeks, _) user_ts tensor and vice-versa    
    '''
    def __init__(self, users_ts: np.ndarray, users_ts_ids: np.ndarray, date_ind, pred_model, max_trans: int):

        self.pred_model = pred_model
        self.max_trans = max_trans

        self.users_ts = users_ts
        self.users_ts_ids = users_ts_ids
        self.date_ind = date_ind

    def __getitem__(self, key: int):
        'returns both predicted and actual output for a customer not that important'
        return None

    def get_predictions_and_actual_transactions(self, start_prediction: str, end_prediction: str) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Given a start_prediction date and an end_prediction date returns both the actual and predicted amount of 
        transactions for each customer in form of two tensors with dim (customers, weeks, features)
        Returns:
            - Actual amount of transactions of users per week; dim (customer, week)
            - Predicted amount of transactions of users per week (customer, week)
        '''
        inputs, preds = self._predict_future_transactions(start_prediction, end_prediction)

        #predicted timeline
        inputs_num = self._one_hot_tensor_to_num(inputs, self.max_trans)
        pred_users_trans_nums = torch.cat((inputs_num, preds), dim=1)

        #actual timeline
        actual_users_trans_nums = self._one_hot_tensor_to_num(self.users_ts, self.max_trans)
        
        return actual_users_trans_nums, pred_users_trans_nums
    

    def _predict_future_transactions(
            self, start_prediction: str, end_prediction: str, batch_size: int = 32
        ) -> Tuple[np.ndarray, torch.Tensor]:
        '''uses the transaction data in time period [0, ind(start_prediction) - 1] 
        and predicts the time period [start_prediction, end_prediction]
        
        Returns:
            input_ts used to predict future transactions; time period [0, ind(start_prediction) - 1]
            predicted future transactions as a Tensor with dimension (customer, week, features);
                time_period [ind(start_prediction), ind(end_prediction)] 
        '''
        # Calculate indices
        pred_start_ind = self.date_ind[start_prediction]
        pred_end_ind = self.date_ind[end_prediction]
        pred_length = pred_end_ind - pred_start_ind + 1

        # Model configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pred_model.to(device)
        self.pred_model.set_label_specs(pred_length, pred_length)
        self.pred_model.eval()

        # Prepare the input tensor for the batch of users
        input_tensor = self.users_ts[:, :pred_start_ind, :]
        num_users = input_tensor.shape[0]
        preds_list = []

        for i in range(0, num_users, batch_size):
            batch_input_tensor = torch.tensor(
                input_tensor[i:i+batch_size], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                batch_preds = self.pred_model(batch_input_tensor)
            batch_preds = self._trans_probs_to_exp_val(batch_preds, self.max_trans)
            preds_list.append(batch_preds.cpu())

        preds = torch.cat(preds_list, dim=0)
        return input_tensor, preds
    
    def _one_hot_tensor_to_num(self, trans_array: torch.Tensor, max_trans: int) -> torch.Tensor:
        '''transforms the transaction tensors using one-hot-encoding to tensors with only an integer
        signifying the amount of transactions happening in one week
        '''
        #dropping one-hot encoding of week
        trans_array = trans_array[:, :, :(max_trans + 1)]
        trans_array = np.argmax(trans_array, axis=2)
        return torch.tensor(trans_array)
    
    def _trans_probs_to_exp_val(self, trans_tensor: torch.Tensor, max_trans: int) -> torch.Tensor:
        '''transforms the transaction tensors using one-hot-encoding to tensors with the expected value
        of the transactions happening in one week
        '''
        # Dropping one-hot encoding of week
        trans_probs = F.softmax(trans_tensor, dim=2)
        
        # Create a tensor of transaction values (0, 1, 2, ..., max_trans)
        trans_values = torch.arange(max_trans + 1, dtype=trans_probs.dtype, device=trans_probs.device)
        
        # Compute the expected value
        expected_values = torch.sum(trans_probs * trans_values, dim=2)
        
        return expected_values