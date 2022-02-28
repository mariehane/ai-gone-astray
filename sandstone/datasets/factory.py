import pickle
from sandstone.utils.generic import log
NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]

# Depending on arg, build dataset
def get_dataset(args):

    dataset_class = get_dataset_class(args)
    args.exam_to_year_dict = {}
    args.exam_to_device_dict = {}

    train = dataset_class(args, 'train')
    dev = dataset_class(args, 'dev')
    test = dataset_class(args, 'test')

    return train, dev, test

# build just one dataset, given split = ['train', 'dev', 'test']
def get_specific_dataset(args, split):
    dataset_class = get_dataset_class(args)
    data = dataset_class(args, split)

    return data