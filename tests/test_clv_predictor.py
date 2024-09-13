import torch

from clv_prediction import clv_predictor

# def test_one_hot_tensor_to_num():
#     tensor_torch = torch.tensor([[
#         [0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0]],

#         [
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0]]])

#     predictor = clv_predictor.CLVPredictor(0, None, None, None)
#     tensor_num = predictor._trans_probs_to_exp_val(tensor_torch, 7)
#     assert torch.equal(tensor_num, torch.Tensor([[1, 2, 5], [4, 6, 1]])) 