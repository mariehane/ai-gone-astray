from torch import nn
import pdb
from sandstone.models.factory import RegisterModel

@RegisterModel("hiddens_mlp")
class MLP(nn.Module):
    """A simple MLP for recursion task"""
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        model_layers = []
        cur_dim = args.input_dim
        for layer in range(args.num_layers):
            bn = nn.BatchNorm1d(cur_dim)
            linear_layer = nn.Linear(cur_dim, args.hidden_dim)
            cur_dim = args.hidden_dim
            model_layers.extend( [bn, linear_layer, nn.ReLU()])

        self.encoder = nn.Sequential(*model_layers)

        self.bn_final = nn.BatchNorm1d(cur_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc_final =nn.Linear(cur_dim, args.num_classes)

        self.pred_activ_assay = args.pred_drug_activ and 'cell_painter' in args.dataset
        if self.pred_activ_assay:
            self.pred_activ_assay = True
            num_assays = len(list(args.drug_to_y.values())[0]['active'])
            self.assay_ffn = nn.Sequential(nn.Linear(cur_dim, args.hidden_dim),
                                            nn.BatchNorm1d(args.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(args.hidden_dim, num_assays)
                                    )


    def forward(self, x, batch=None):
        x = x.view( x.size()[0], -1)
        output = {'hidden': self.encoder(x)}
        output['logit'] = self.fc_final(self.dropout(self.bn_final(output['hidden'])))
        if self.pred_activ_assay:
            output['drug_activ_logit'] = self.assay_ffn(output['hidden'])
        return output

