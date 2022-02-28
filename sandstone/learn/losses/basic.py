from sandstone.learn.losses.factory import RegisterLoss
import torch
import torch.nn.functional as F
from collections import OrderedDict

@RegisterLoss("cross_entropy")
def get_model_loss(model_output, batch, model, args):
    logging_dict, predictions = OrderedDict(), OrderedDict()
    logit = model_output['logit']
    loss = F.cross_entropy(logit, batch['y'].long())
    logging_dict['cross_entropy_loss'] = loss.detach()
    predictions['probs'] = F.softmax(logit, dim=-1).detach()
    return loss, logging_dict, predictions
