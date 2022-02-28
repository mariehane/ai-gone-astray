import torch
from sandstone.datasets.mimic_iv import MIMIC_IV_Abstract_Dataset

def get_linear_feature_weights(args, model, dataset: MIMIC_IV_Abstract_Dataset):
    weights, bias = model.model.parameters()

    weights = weights[1]
    odds = torch.exp(weights)

    features = dataset.static_features

    return features, weights, odds

