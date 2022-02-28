import torch
from torch import nn
from sandstone.models.factory import RegisterModel
from sandstone.models.pools.factory import get_pool

class AbstractRNN(nn.Module):
    """Base class for RNNs on timeseries inputs.
    Feeds static inputs through a seperate MLP and then concats outputs 
    before feeding through a final fully-connected layer"""
    def __init__(self, args):
        super(AbstractRNN, self).__init__()
        self.args = args

        # rnn for timeseries inputs
        self.rnn_encoder = self.get_rnn(self.args, batch_first=True)

        rnn_output_dim = args.rnn_hidden_dim
        if args.rnn_bidirectional:
            rnn_output_dim *= 2

        if args.rnn_pool_name is not None:
            self.rnn_pool = get_pool(self.args.rnn_pool_name)(args, rnn_output_dim)
        
        # fully-connected for static
        fc_layers = []
        cur_dim = args.input_dim
        for layer in range(args.num_layers):
            bn = nn.BatchNorm1d(cur_dim)
            linear_layer = nn.Linear(cur_dim, args.hidden_dim)
            cur_dim = args.hidden_dim
            fc_layers.extend( [bn, linear_layer, nn.ReLU()] )
        self.fc_encoder = nn.Sequential(*fc_layers)

        final_fc_dim = rnn_output_dim + args.hidden_dim

        self.bn_final = nn.BatchNorm1d(final_fc_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc_final = nn.Linear(final_fc_dim, args.num_classes)

    def forward(self, x, batch=None):
        # pass timeseries vars through rnn
        x_timeseries = batch['x_timeseries']
        rnn_output, _ = self.rnn_encoder(x_timeseries)

        if self.args.rnn_pool_name is None:
            # select last hidden state
            rnn_output = rnn_output[:, -1]
        else:
            rnn_output = self.rnn_pool(rnn_output)

        # pass static vars through fc
        fc_hidden = self.fc_encoder(x)
        
        # concatenate with static vars to feed to final fully-connected layer
        hidden = torch.cat((rnn_output, fc_hidden), dim=-1)
        output = {'hidden': hidden}

        output['logit'] = self.fc_final(self.dropout(self.bn_final(hidden)))
        return output
    
    def get_rnn(self, args, batch_first):
        raise NotImplementedError

@RegisterModel("rnn")
class RNN(AbstractRNN):
    """A basic Elman RNN"""

    def get_rnn(self, args, batch_first):
        return nn.RNN(args.timeseries_dim, args.rnn_hidden_dim, args.rnn_num_layers, 
                      batch_first=batch_first, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)

@RegisterModel("lstm")
class LSTM(AbstractRNN):
    """Long Short Term Memory RNN"""

    def get_rnn(self, args, batch_first):
        return nn.LSTM(args.timeseries_dim, args.rnn_hidden_dim, args.rnn_num_layers, 
                       batch_first=batch_first, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)

@RegisterModel("gru")
class GRU(AbstractRNN):
    """Gated Recurrent Unit"""

    def get_rnn(self, args, batch_first):
        return nn.GRU(args.timeseries_dim, args.rnn_hidden_dim, args.rnn_num_layers, 
                      batch_first=batch_first, dropout=args.rnn_dropout, bidirectional=args.rnn_bidirectional)

