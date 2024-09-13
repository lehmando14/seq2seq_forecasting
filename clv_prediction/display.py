import torch
import matplotlib.pyplot as plt

def display_for_dataset(actual_trans: torch.Tensor, pred_trans: torch.Tensor, date_indexer, prediction_start: str) -> None:
    '''used to display the actual aggregated transactions'''
    actual_trans = actual_trans.numpy()

    pred_trans = pred_trans.numpy()
    pred_trans_y = pred_trans[date_indexer[prediction_start]:]
    pred_trans_x = list(range(
        date_indexer[prediction_start], 
        date_indexer[prediction_start] + len(pred_trans_y)))

    plt.figure(figsize=(12,8))

    # plt.plot(pred_trans_x, pred_trans_y, color='blue')
    plt.plot(actual_trans, color='black')
    plt.axvline(x=date_indexer[prediction_start], color='red', linestyle='--')

    # Add text label next to the vertical line
    plt.text(date_indexer[prediction_start] + 1, 
             max(actual_trans) * 0.95, 
             'Start Prediction Period', 
             color='red', 
             verticalalignment='top')

    plt.xlabel('Period (Week)')
    plt.ylabel('Aggregated Transactions')
    plt.show()
    return None

def display(actual_trans: torch.Tensor, pred_trans: torch.Tensor, date_indexer, prediction_start: str) -> None:
    actual_trans = actual_trans.numpy()

    pred_trans = pred_trans.numpy()
    pred_trans_y = pred_trans[date_indexer[prediction_start]:]
    pred_trans_x = list(range(
        date_indexer[prediction_start], 
        date_indexer[prediction_start] + len(pred_trans_y)))

    plt.figure(figsize=(12,8))

    plt.plot(pred_trans_x, pred_trans_y, color='blue')
    plt.plot(actual_trans, color='black')
    plt.axvline(x=date_indexer[prediction_start], color='red', linestyle='--')


    plt.xlabel('Period (Weeks)')
    plt.ylabel('Aggregated Transactions')
    plt.title('Aggregated Transactions Over Time')
    plt.show()
    return None

def display_three(actual_trans: torch.Tensor, pred_trans1: torch.Tensor, pred_trans2: torch.Tensor, date_indexer, prediction_start: str) -> None:
    '''compares to forecasts to the actual values'''
    actual_trans = actual_trans.numpy()
    
    pred_trans1 = pred_trans1.numpy()
    pred_trans2 = pred_trans2.numpy()
    
    pred_trans1_y = pred_trans1[date_indexer[prediction_start]:]
    pred_trans2_y = pred_trans2[date_indexer[prediction_start]:]
    
    pred_trans_x = list(range(
        date_indexer[prediction_start], 
        date_indexer[prediction_start] + len(pred_trans1_y)))
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(pred_trans_x, pred_trans1_y, color='blue', label='Seq2Seq')
    plt.plot(pred_trans_x, pred_trans2_y, color='green', label='PNBD')
    plt.plot(actual_trans, color='black', label='Actual')
    
    plt.axvline(x=date_indexer[prediction_start], color='red', linestyle='--', label='Prediction Start')
    
    plt.xlabel('Period (Week)')
    plt.ylabel('Aggregated Transactions')
    plt.legend()
    plt.show()

    return None

def display_prediction_period(actual_trans: torch.Tensor, pred_trans: torch.Tensor, date_indexer, prediction_start: str) -> None:
    actual_trans = actual_trans.numpy()
    actual_trans_y = actual_trans[date_indexer[prediction_start]:]

    pred_trans = pred_trans.numpy()
    pred_trans_y = pred_trans[date_indexer[prediction_start]:]

    trans_x = list(range(
        date_indexer[prediction_start], 
        date_indexer[prediction_start] + len(pred_trans_y)))
    

    plt.figure(figsize=(12,8))

    plt.plot(trans_x, pred_trans_y, color='blue')
    plt.plot(trans_x, actual_trans_y, color='black')
    plt.axvline(x=date_indexer[prediction_start], color='red', linestyle='--')

    plt.xlabel('Period (Weeks)')
    plt.ylabel('Aggregated Transactions')
    plt.show()
    return None

def display_prediction_period_three(actual_trans: torch.Tensor, pred_trans1: torch.Tensor, pred_trans2: torch.Tensor, date_indexer, prediction_start: str) -> None:
    actual_trans = actual_trans.numpy()
    actual_trans_y = actual_trans[date_indexer[prediction_start]:]

    pred_trans1 = pred_trans1.numpy()
    pred_trans1_y = pred_trans1[date_indexer[prediction_start]:]

    pred_trans2 = pred_trans2.numpy()
    pred_trans2_y = pred_trans2[date_indexer[prediction_start]:]

    trans_x = list(range(
        date_indexer[prediction_start], 
        date_indexer[prediction_start] + len(pred_trans1_y)))

    plt.figure(figsize=(12, 8))

    plt.plot(trans_x, pred_trans1_y, color='blue', label='Seq2Seq')
    plt.plot(trans_x, pred_trans2_y, color='green', label='PNBD')
    plt.plot(trans_x, actual_trans_y, color='black', label='Actual')

    plt.axvline(x=date_indexer[prediction_start], color='red', linestyle='--', label='Prediction Start')

    plt.xlabel('Period (Week)')
    plt.ylabel('Aggregated Transactions')
    plt.legend()
    plt.show()

    return None