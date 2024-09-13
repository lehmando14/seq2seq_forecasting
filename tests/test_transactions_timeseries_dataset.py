from preprocessing.transactions_timeseries_dataset import TransactionsTimeseriesDataset
import numpy as np

def test_simple_window():
    users_trans_ts =  np.array([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 8]],
                            [[0, 0], [0, 1], [0, 2], [1, 3], [0, 4], [0, 5], [1, 6], [0, 7], [2, 8]],
                            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 7], [0, 8]]])

    ttd = TransactionsTimeseriesDataset(users_trans_ts, -1, 3, 1, 1, 1)

    assert len(ttd) == 18

    #test 1
    input_ts, label_ts = ttd[0]
    assert np.array_equal(input_ts, np.array([[0, 0], [0, 1], [0, 2]]))
    assert np.array_equal(label_ts, np.array([[0, 3]]))

    #test 2
    input_ts, label_ts = ttd[7]
    assert np.array_equal(input_ts, np.array([[0, 1], [0, 2], [1, 3]]))
    assert np.array_equal(label_ts, np.array([[0, 4]]))

def test_complex_window():
    users_trans_ts =  np.array([[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 8]],
                         [[0, 0], [0, 1], [0, 2], [1, 3], [0, 4], [0, 5], [1, 6], [0, 7], [2, 8]],
                         [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 7], [0, 8]]])

    ttd = TransactionsTimeseriesDataset(users_trans_ts, -1, 4, 3, 2, 3)

    assert len(ttd) == 6

    #test 1
    input_ts, label_ts = ttd[1]
    assert np.array_equal(input_ts, np.array([[0, 3], [0, 4], [0, 5], [0, 6]]))
    assert np.array_equal(label_ts, np.array([[0, 6], [0, 7], [1, 8]]))

    # #test 2
    input_ts, label_ts = ttd[5]
    assert np.array_equal(input_ts, np.array([[0, 3], [0, 4], [0, 5], [0, 6]]))
    assert np.array_equal(label_ts, np.array([[0, 6], [1, 7], [0, 8]]))

def test_newest_version_window():
    users_trans_ts =  np.array([
                        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]],
                        [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]]])
    
    dataset = TransactionsTimeseriesDataset(users_trans_ts, 1, 3, 1, 1, 1)
    input_ts, label_ts = dataset[1]
    assert np.array_equal(input_ts, np.array([[0., 1., 0.],   # Sample 1
                        [1., 0., 0.],   # Sample 2
                        [0., 1., 0.]], dtype=np.float32))
    assert np.array_equal(label_ts, np.array([[1., 0.]], dtype=np.float32) )