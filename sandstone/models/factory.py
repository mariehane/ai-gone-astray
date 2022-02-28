import torch
from torch import nn
from sandstone.utils.generic import log
MODEL_REGISTRY = {}

STRIPPING_ERR = 'Trying to strip the model although last layer is not FC.'
NO_MODEL_ERR = 'Model {} not in MODEL_REGISTRY! Available models are {} '
NO_OPTIM_ERR = 'Optimizer {} not supported!'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with (block_name, num_repeats). Received {}'
NUM_MATCHING_LAYERS_MESSAGE = 'Loaded pretrained_weights for {} out of {} parameters.'

def RegisterModel(model_name):
    """Registers a configuration."""

    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(args):
    return get_model_by_name(args.model_name, args)


def get_model_by_name(name, args):
    '''
        Get model from MODEL_REGISTRY based on args.model_name
        args:
        - name: Name of model, must exit in registry
        - allow_wrap_model: whether or not override args.wrap_model and disable model_wrapping.
        - args: run ime args from parsing

        returns:
        - model: an instance of some torch.nn.Module
    '''
    if not name in MODEL_REGISTRY:
        raise Exception(
            NO_MODEL_ERR.format(
                name, MODEL_REGISTRY.keys()))


    model = MODEL_REGISTRY[name](args)
    allow_data_parallel = 'discriminator' not in name and ('mirai_full' not in args.model_name)
    return wrap_model(model, args, allow_data_parallel)

def wrap_model(model, args, allow_data_parallel=True):

    if args.state_dict_path is not None:
        load_pretrained_weights(model, torch.load(args.state_dict_path), args)

    return model

def load_model(path, args):
    log('\nLoading model from [%s]...' % path, args)
    try:
        model = torch.load(path, map_location=args.map_location)
    except:
        raise Exception(
            "Sorry, snapshot {} does not exist!".format(path))

    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module.cpu()
    if hasattr(model, 'args'):
        model.args = args
    return model

def validate_block_layout(block_layout):
    """Confirms that a block layout is in the right format.

    Arguments:
        block_layout(list): A length n list where each of the n elements
         is a list of lists where each inner list is of length 2 and
         contains (block_name, num_repeats). This specifies the blocks
         in each of the n layers of the ResNet.

    Raises:
        Exception if the block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. (block_name, num_repeats))
    for layer_layout in block_layout:
        for block_spec in layer_layout:
            if len(block_spec) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(block_spec))


def get_optimizer(model, args):
    '''
    Helper function to fetch optimizer based on args.
    '''
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == 'adam':
        optimizer =  torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer =  torch.optim.SGD(params,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum )
    else:
        raise Exception(NO_OPTIM_ERR.format(args.optimizer))

    scheduler =  {
         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                        patience= args.patience, factor= args.lr_decay,\
                        mode = 'min' if 'loss' in args.tuning_metric else 'max'),
         'monitor': 'val_{}'.format(args.tuning_metric),
         'interval': 'epoch',
         'frequency': 1
      }
    return [optimizer], [scheduler]


def load_pretrained_weights(model, pretrained_state_dict, args):
    """Loads pretrained weights into a model (even if not all layers match).

    Arguments:
        model(Model): A PyTorch model.
        pretrained_state_dict(dict): A dictionary mapping layer names
            to pretrained weights.
    """
    model_state_dict = model.state_dict()

    # Filter out pretrained layers not in our model
    matching_pretrained_state_dict = {
        layer_name: weights
        for layer_name, weights in pretrained_state_dict.items()
        if (layer_name in model_state_dict and
            pretrained_state_dict[layer_name].size() == model_state_dict[layer_name].size())
    }

    log(NUM_MATCHING_LAYERS_MESSAGE.format(len(matching_pretrained_state_dict),
                                             len(model_state_dict)), args)
    # Overwrite weights in existing state dict
    model_state_dict.update(matching_pretrained_state_dict)

    # Load the updated state dict
    model.load_state_dict(model_state_dict)



