import torch
from collections import OrderedDict
from sandstone.learn.losses.factory import RegisterLoss

@RegisterLoss("l1_regularization")
def get_model_loss(model_output, batch, lightning_module, args):
    """Computes l1 regularization term
    
    Note: only works for linear model
    """
    logging_dict, predictions = OrderedDict(), OrderedDict()

    model = lightning_module.model
    weights, bias = model.parameters()

    loss = weights.abs().sum()

    return loss * args.l1_regularization_lambda, logging_dict, predictions

