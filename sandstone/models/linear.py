from torch import nn
import pdb
from sandstone.models.factory import RegisterModel

@RegisterModel("linear")
class Linear(nn.Module):
    """A super simple, easily interpretable linear model for regression/classification"""
    def __init__(self, args):
        super(Linear, self).__init__()
        self.args = args
        self.model = nn.Linear(args.input_dim, args.num_classes)

    def forward(self, x, batch=None):
        x = x.view(x.size()[0], -1)
        output = {
            'logit': self.model(x)
        }
        return output
    
    def get_loss_functions(self, args):
        loss_fns = ['mse'] if args.num_classes == 1 \
                    else ['cross_entropy']

        if args.l1_regularization_lambda > 0:
            loss_fns.append('l1_regularization')

        return loss_fns

