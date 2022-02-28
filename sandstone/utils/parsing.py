import argparse
import os
import pwd
from pytorch_lightning import Trainer

EMPTY_NAME_ERR = 'Name of augmentation or one of its arguments cant be empty\n\
                  Use "name/arg1=value/arg2=value" format'
BATCH_SIZE_SPLIT_ERR = 'batch_size (={}) should be a multiple of batch_splits (={})'
INVALID_IMG_TRANSFORMER_SPEC_ERR = 'Invalid image transformer embedding args. Must be length 3, as [name/size=value/dim=value]. Received {}'
INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR = 'Image transformer embeddings have different embedding dimensions {}'
INVALID_NUM_BLOCKS_ERR = 'Invalid block_layout. Must be length 4. Received {}'
INVALID_BLOCK_SPEC_ERR = 'Invalid block specification. Must be length 2 with "block_name,num_repeats". Received {}'
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'
INVALID_DATASET_FOR_SURVIVAL = "A dataset with '_full_future'  can only be used with survival_analysis_setup and viceversa."
NPZ_MULTI_IMG_ERROR = "Npz loading code assumes multi images are in one npz and code is only in multi-img code flow."
SELF_SUPER_ERROR = "Moco and Byol only supported with instance disrimination task. Must be multi image with 2 images"
PRINT_LINEAR_INVALID_MODEL = "--print_linear_weights can only be used with the linear model!"
TIMESERIES_SUMMARY_ERROR = "Only one of --flatten_timeseries and --timeseries_moments can be specified at a time!"

def parse_augmentations(raw_augmentations):
    """
    Parse the list of augmentations, given by configuration, into a list of
    tuple of the augmentations name and a dictionary containing additional args.

    The augmentation is assumed to be of the form 'name/arg1=value/arg2=value'

    :raw_augmentations: list of strings [unparsed augmentations]
    :returns: list of parsed augmentations [list of (name,additional_args)]

    """
    augmentations = []
    for t in raw_augmentations:
        arguments = t.split('/')
        name = arguments[0]
        if name == '':
            raise Exception(EMPTY_NAME_ERR)

        kwargs = {}
        if len(arguments) > 1:
            for a in arguments[1:]:
                splited = a.split('=')
                var = splited[0]
                val = splited[1] if len(splited) > 1 else None
                if var == '':
                    raise Exception(EMPTY_NAME_ERR)

                kwargs[var] = val

        augmentations.append((name, kwargs))

    return augmentations

def parse_embeddings(raw_embeddings):
    """
    Parse the list of embeddings, given by configuration, into a list of
    tuple of the embedding embedding_name, size ('vocab size'), and the embedding dimension.

    :raw_embeddings: list of strings [unparsed transformers], each of the form 'embedding_name/size=value/dim=value'
    :returns: list of parsed embedding objects [(embedding_name, size, dim)]

    For example:
        --hidden_transformer_embeddings time_seq/size=10/dim=32 view_seq/size=2/dim=32 side_seq/size=2/dim=32
    returns
        [('time_seq', 10, 32), ('view_seq', 2, 32), ('side_seq', 2, 32)]
    """
    embeddings = []
    for t in raw_embeddings:
        arguments = t.split('/')
        if len(arguments) != 3:
                raise Exception(INVALID_IMG_TRANSFORMER_SPEC_ERR.format(len(arguments)))
        name = arguments[0]
        size = arguments[1].split('=')[-1]
        dim = arguments[2].split('=')[-1]

        embeddings.append((name, int(size), int(dim)))

    if not all([embed[-1] == int(dim) for embed in embeddings]):
        raise Exception(INVALID_IMG_TRANSFORMER_EMBED_SIZE_ERR.format([embed[-1] for embed in embeddings]))
    return embeddings

def validate_raw_block_layout(raw_block_layout):
    """Confirms that a raw block layout is in the right format.

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout in the format
            'block_name,num_repeats-block_name,num_repeats-...'

    Raises:
        Exception if the raw block layout is formatted incorrectly.
    """

    # Confirm that each layer is a list of block specifications where
    # each block specification has length 2 (i.e. block_name,num_repeats)
    for raw_layer_layout in raw_block_layout:
        for raw_block_spec in raw_layer_layout.split('-'):
            if len(raw_block_spec.split(',')) != 2:
                raise Exception(INVALID_BLOCK_SPEC_ERR.format(raw_block_spec))


def parse_block_layout(raw_block_layout):
    """Parses a ResNet block layout, which is a list of layer layouts
    with each layer layout in the form 'block_name,num_repeats-block_name,num_repeats-...'

    Example:
        ['BasicBlock,2',
         'BasicBlock,1-NonLocalBlock,1',
         'BasicBlock,3-NonLocalBlock,2-Bottleneck,2',
         'BasicBlock,2']
        ==>
        [[('BasicBlock', 2)],
         [('BasicBlock', 1), ('NonLocalBlock', 1)],
         [('BasicBlock', 3), ('NonLocalBlock', 2), ('Bottleneck', 2)],
         [('BasicBlock', 2)]]

    Arguments:
        raw_block_layout(list): A list of strings where each string
            is a layer layout as described above.

    Returns:
        A list of lists of length 4 (one for each layer of ResNet). Each inner list is
        a list of tuples, where each tuple is (block_name, num_repeats).
    """

    validate_raw_block_layout(raw_block_layout)

    block_layout = []
    for raw_layer_layout in raw_block_layout:
        raw_block_specs = raw_layer_layout.split('-')
        layer = [raw_block_spec.split(',') for raw_block_spec in raw_block_specs]
        layer = [(block_name, int(num_repeats)) for block_name, num_repeats in layer]
        block_layout.append(layer)

    return block_layout


def parse_dispatcher_config(config):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item,
    but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config

    returns: jobs - a list of flag strings, each of which encapsulates one job.
        *Example: --train --cuda --dropout=0.1 ...
    returns: experiment_axies - axies that the grid search is searching over
    '''
    jobs = [""]
    experiment_axies = []
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]


    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):
            possible_values = search_space[flag]
            if len(possible_values) > 1:
                experiment_axies.append(flag)

            children = []
            if len(possible_values) == 0 or type(possible_values) is not list:
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))
            for value in possible_values:
                for parent_job in jobs:
                    if type(value) is bool:
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    elif type(value) is list:
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(parent_job, flag,
                                                          val_list_str)
                    else:
                        new_job_str = "{} --{} {}".format(parent_job, flag, value)
                    children.append(new_job_str)
            jobs = children

    return jobs, experiment_axies

def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description='EHR Model Drift Analysis')
    # setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    parser.add_argument('--dev', action='store_true', default=False, help='Whether or not to run model on dev set')
    parser.add_argument('--eval_train', action='store_true', default=False, help='Whether or not to evaluate model on train set')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='Whether or not to fine_tune model')
    parser.add_argument('--num_epochs_fine_tune', type=int, default=1, help='Num epochs to finetune model')
    parser.add_argument('--lightning_name', type=str, default='default', help="Name of lightning module to structure training.")

    # data
    parser.add_argument('--dataset', default='mnist', help='Name of dataset from dataset factory to use [default: mnist]')

    parser.add_argument('--num_workers', type=int, default=8, help='num workers for each data loader [default: 4]')
    parser.add_argument('--drop_last', action='store_true', default=False, help='Whether to drops the last non-full batch. Prevents errors with low batch-sizes for e.g. batch-normalization.')

    parser.add_argument('--input_dim', type=int, default=512, help='Input dim for 2stage models. [default:512]')
    parser.add_argument('--cache_path', type=str, default=None, help='dir to cache images.')

    # sampling
    parser.add_argument('--class_bal', action='store_true', default=False, help='Wether to apply a weighted sampler to balance between the classes on each batch.')
    parser.add_argument('--split_probs', type=float, nargs='+', default=[0.6, 0.2, 0.2], help='Split probs for datasets without fixed train dev test. ')
    # rationale

    # regularization
    parser.add_argument('--l1_regularization_lambda',  type=float, default=0,  help='L1 regularization penalty.')
    parser.add_argument('--hidden_size', type=int, default=300,
                    help='Dimensionality of hidden layers in MPN')

    # MIMIC-IV / Sepsis
    parser.add_argument('--dataset_path', type=str, default='/data/mimiciv', help='Path to dataset')
    parser.add_argument('--item_map_path', type=str, help='Path to csv with itemids to keep')
    parser.add_argument('--vent_path', type=str, default='data/ventilation.csv.gz', help='Path to ventilation.csv.gz')
    parser.add_argument('--task', type=str, choices=['sepsis3', 'los', 'icumort'], help='Prediction label (los = long length of stay).')
    parser.add_argument('--nrows', type=int, default=None,  help='No. of rows to truncate data to')
    parser.add_argument('--chunksize', type=int, default=None,  help='No. of rows to load at once')
    parser.add_argument('--flatten_timeseries', action='store_true',  default=False, help='Whether to flatten the hourly values along the time dimension')
    parser.add_argument('--timeseries_moments', type=str, nargs='+', default=None, required=False, choices=['mean', 'var', 'skew', 'kurt'], help='Whether to calculate moments over the timeseries for each feature.')
    parser.add_argument('--min_patient_age', type=int, default=15,  help='Min. patient age to include')
    parser.add_argument('--data_hours', type=int, default=24,  help='Min. hours of patient hospital stay to include.')
    parser.add_argument('--min_hours', type=int, default=24,  help='Min. hours of patient hospital stay to include.')
    parser.add_argument('--gap_hours', type=int, default=6,  help='Min. no. of hours betwween data and prediction label. Used to prevent label leakage.')
    parser.add_argument('--min_icu_stay', type=int, default=12,  help='Min. length of patient stay in ICU to include')
    parser.add_argument('--max_icu_stay', type=int, default=10,  help='Max. length of patient stay in ICU to include')
    parser.add_argument('--train_years', type=str, default=None, help='Years to train on for time drift, inclusive')
    parser.add_argument('--test_years', type=str, nargs='+', default=[], help='Years to test on for time drift, inclusive')
    parser.add_argument('--train_task', type=str, default='sepsis3', help='Sepsis definition to train on')
    parser.add_argument('--test_tasks', type=str, nargs='+', default=['sepsis3'], help='All definitions to test on')
    # parser.add_argument('--multiple_tests', action='store_true', default=False, help='Whether there are multiple testing buckets to run')
    parser.add_argument('--sepsis_consider_sofa_difference', action='store_true', default=False, help='Whether to use sepsis calculation that takes increases from baseline into account')
    parser.add_argument('--sepsis_decrease_sofa_baseline', action='store_true', default=False, help='Whether the SOFA baseline is allowed to decrease over time if a lower SOFA is encountered')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for EHR data, if applicable')
    parser.add_argument('--cache_static_filename', type=str, default="static_df.csv", help='Name of static data file to retrieve or save from cache')
    parser.add_argument('--cache_hourly_filename', type=str, default="hourly_df.csv", help='Name of hourly aggregated data file to retrieve or save from cache')
    parser.add_argument('--timesplit', action='store_true', default=False, help='Whether to run timesplit experiment')
    parser.add_argument('--print_linear_weights', action='store_true', default=False, help='Whether to print feature importances for linear model')
    parser.add_argument('--features', type=str, nargs='+', default=[], help='Level 2 Feature names to test on. When default empty, uses all features in item_map')
    parser.add_argument('--feature_search', type=str, default=None, help='Individual feature to include in addition to the --features')
    parser.add_argument('--feature_remove', type=str, nargs='+', default=[], help='Level 2 Feature name(s) to remove from item map path')
    parser.add_argument('--case_control', action='store_true', default=False, help='Whether to implement control case balancing')
    parser.add_argument('--dascena_control', action='store_true', default=False, help='Whether to use dascena form of random control matching')
    parser.add_argument('--dascena_drop', action='store_true', default=False, help='Whether to drop patients who have one or more dascena vitals missing')
    parser.add_argument('--group_by_level2', action='store_true', default=False, help='Whether to group itemids together by their level 2 label')
    parser.add_argument('--normalize', type=str, nargs='+', default=None, help='If set, normalize the given features')
    parser.add_argument('--los_threshold', type=int, default=4, help='Threshold in days for length of stay label')
    parser.add_argument('--impute_method', type=str, default='simple', help='Method for imputation, specify zeroes, advanced, or simple')
    parser.add_argument('--test_onset_buckets', action='store_true', default=False, help='Whether to test AUC based on sepsis onset hours')
    parser.add_argument('--onset_buckets', type=str, nargs='+', default=[], help='Hours to test as onset buckets')

    # learning
    parser.add_argument('--optimizer', type=str, default="adam", help='optimizer to use [default: adam]')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum to use with SGD')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='initial learning rate [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 Regularization penaty [default: 0]')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs without improvement on dev before halving learning rate and reloading best model [default: 5]')
    parser.add_argument('--tuning_metric', type=str, default='loss', help='Metric to judge dev set results. Possible options include auc, loss, accuracy [default: loss]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training [default: 128]')
    parser.add_argument('--dropout', type=float, default=0.25, help='Amount of dropout to apply on last hidden layer [default: 0.25]')

    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to dump the model')
    parser.add_argument('--results_path', type=str, default='logs/test.args', help='where to save the result logs')

    parser.add_argument('--project_name', type=str, default=None, help='Name of project for comet logger')
    parser.add_argument('--workspace', type=str, default=None, help='Name of workspace for comet logger')
    parser.add_argument('--comet_tags', nargs='*', default=[], help="List of tags for comet logger")
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use, i.e 1.0 for all and 0 for none. Used for learning curve analysis.')

    # Alternative training/testing schemes
    parser.add_argument('--cross_val_seed', type=int, default=0, help="Seed used to generate the partition.")
    parser.add_argument('--trainer_name', type=str, default='default', help="Form of model, i.e resnet18, aggregator, revnet, etc.")
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    parser.add_argument('--state_dict_path', type=str, default=None, help='filename of model snapshot to load[default: None]')

    # model configuration / hyperparams
    parser.add_argument('--model_name', type=str, default='linear', help="Form of model, i.e linear, mlp, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=512, help='num neurons in hidden layers of MLP.')
    parser.add_argument('--num_layers', type=int, default=3, help="num layers for MLP.")

    # RNN
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='hidden dim for recurrent models')
    parser.add_argument('--rnn_num_layers', type=int, default=3, help="Num layers for recurrent models.")
    parser.add_argument('--rnn_dropout', type=float, default=0.25, help='Amount of dropout for recurrent models [default: 0.25]')
    parser.add_argument('--rnn_bidirectional', action='store_true', default=False, help='Makes the rnn bidirectional')
    parser.add_argument('--rnn_pool_name', type=str, default=None, help='Pooling layer to apply to rnn outputs')

    # run
    parser = Trainer.add_argparse_args(parser)
    if args_strings is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_strings)
    args.lr = args.init_lr


    if args.flatten_timeseries or args.timeseries_moments is not None:
        args.static_only = True
    else:
        args.static_only = False

    if args.workspace is None:
        args.workspace = os.environ.get("COMET_WORKSPACE")

    if args.cache_dir is not None and not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)

    if (isinstance(args.gpus, str) and len(args.gpus.split(",")) > 1) or (isinstance(args.gpus, int) and  args.gpus > 1):
        args.distributed_backend = 'ddp'
        args.replace_sampler_ddp = False

    args.unix_username = pwd.getpwuid( os.getuid() )[0]

    # learning initial state
    args.step_indx = 1

    # Parse list args to appropriate data format
    # Check whether certain args or arg combinations are valid
    validate_args(args)

    return args

def validate_args(args):
    """Checks whether certain args or arg combinations are valid.

    Raises:
        Exception if an arg or arg combination is not valid.
    """

    if args.flatten_timeseries and args.timeseries_moments is not None:
        raise ValueError(TIMESERIES_SUMMARY_ERROR)

    if args.print_linear_weights and args.model_name != 'linear':
        raise ValueError(PRINT_LINEAR_INVALID_MODEL)
