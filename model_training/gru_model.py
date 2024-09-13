import torch
from torch import nn
from torch.nn import functional as F

from torch.profiler import record_function

class GRU(nn.Module):

    def __init__(
            self, 
            hidden_size: int, num_layers: int, 
            max_transactions: int,
        ):
        ''''''
        super().__init__()
        input_size = (max_transactions + 1) + 52

        #nn unit specs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #units
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        fc_units = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        fc_units.append(nn.Linear(hidden_size, max_transactions + 1))
        self.fcs = nn.Sequential(*fc_units)

        #label specifications
        self.max_transactions = max_transactions
        self.label_offset = None
        self.label_size = None    

    def forward(self, x):
        batch_size = x.size(0)
        last_weeks = x[:, -1, self.max_transactions + 1:]
        auto_reg_steps = self.label_offset - 1

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, hn = self.gru(x, h0)

        if auto_reg_steps > 0:
            out_auto_reg, _ = self._predict_autoreg(hn, last_weeks, auto_reg_steps)
            out_tot = torch.cat((out, out_auto_reg), dim=1)
        else:
            out_tot = out

        #trans_scores are fed into softmax to get the likelihood of the 
        #different transaction amounts; we don't compute the softmax on purpose
        #pytorch crossentropy loss doesn't require normalization
        trans_scores = self.fcs(out_tot[:, -self.label_size:, :])   
        return trans_scores
    
    def set_label_specs(self, label_offset: int, label_size: int) -> None:
        '''alter label sizes to change what will be predicted'''
        self.label_offset = label_offset
        self.label_size = label_size
        

    def _predict_autoreg(
            self, hn: torch.Tensor, 
            last_weeks: torch.Tensor, num_predicts: int
        ) -> torch.Tensor:
        '''predicts the outputs autoregressively, i.e. using generated output as new input'''
        out_preds = []
        next_weeks = self._create_next_weeks(last_weeks, num_predicts)

        for i in range(num_predicts):
            
            next_trans_pred = self._context_vectors_to_inputs(hn[-1])
        
            next_input = torch.cat((next_trans_pred, next_weeks[:, i, :]), dim=1)
            next_input = next_input.unsqueeze(1)

            _, hn = self.gru(next_input, hn)
            out_preds.append(hn[-1])
        
        return torch.stack(out_preds, dim=1), hn
    
    def _context_vectors_to_inputs(self, c_vecs: torch.Tensor) -> torch.Tensor:
        '''Takes the context vectors hn and creates the inputs in the next period using sampling'''
        # Compute logits (unnormalized scores) instead of using softmax
        logits = self.fcs(c_vecs)

        dist = torch.distributions.Categorical(logits=logits)
        indices = dist.sample()

        one_hot_inputs = F.one_hot(indices, num_classes=(self.max_transactions + 1)).float()
        return one_hot_inputs
    
    def _create_next_weeks(self, last_weeks: torch.Tensor, num_predicts: int) -> torch.Tensor:
        '''creates the next weeks used as inputs'''
        curr_weeks_num = torch.argmax(last_weeks, dim=1)
        offsets = torch.arange(1, num_predicts + 1).to(last_weeks.device)
        next_weeks_num = (curr_weeks_num.unsqueeze(1) + offsets) % 52

        # Create an empty tensor for one-hot encodings & fill
        next_weeks_one_hot = torch.zeros(next_weeks_num.size(0), next_weeks_num.size(1), 52).to(last_weeks.device)
        next_weeks_one_hot.scatter_(2, next_weeks_num.unsqueeze(-1), 1)
        
        return next_weeks_one_hot