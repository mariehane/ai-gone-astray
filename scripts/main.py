import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from os.path import dirname, realpath
import sys
import git
sys.path.append(dirname(dirname(realpath(__file__))))
import sandstone.datasets.factory as dataset_factory
import sandstone.utils.parsing as parsing
from sandstone.utils.generic import get_train_dataset_loader, get_eval_dataset_loader
from sandstone.utils.feature_importances import get_linear_feature_weights
import pytorch_lightning as pl
import sandstone.learn.lightning.factory as lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import _logger as log


#Constants
DATE_FORMAT_STR = "%Y-%m-%d:%H-%M-%S"

def main(args):
    repo = git.Repo(search_parent_directories=True)
    commit  = repo.head.object
    args.commit = commit.hexsha
    result_path_stem = args.results_path.split("/")[-1].split('.')[0]
    log.info("Main running from commit: \n\n{}\n{}author: {}, date: {}".format(
        commit.hexsha, commit.message, commit.author, commit.committed_date))

    trainer = pl.Trainer.from_argparse_args(args)
    args.num_nodes = trainer.num_nodes
    args.num_processes = trainer.num_processes
    args.world_size = args.num_nodes * args.num_processes
    args.global_rank = trainer.global_rank
    args.local_rank = trainer.local_rank
    args.current_test_years = args.train_years
    args.task = args.train_task
    args.onset_bucket = None

    snapshot_dir = os.path.join(args.save_dir, result_path_stem)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)
    print("snapshot_dir: {}".format(snapshot_dir))
    checkpoint_callback = ModelCheckpoint(
        filepath=snapshot_dir,
        save_top_k=1,
        verbose=True,
        monitor='val_{}'.format(args.tuning_metric),
        mode='min' if 'loss' in args.tuning_metric else 'max',
        prefix=""
    )
    checkpoint_callback = trainer.callback_connector.init_default_checkpoint_callback(checkpoint_callback)

    trainer.checkpoint_callback = checkpoint_callback
    for i, callback in enumerate(trainer.callbacks):
        if isinstance(callback, ModelCheckpoint):
            trainer.callbacks[i] = checkpoint_callback
    # Load lightning module and trainer
    tb_logger = pl.loggers.CometLogger(api_key=os.environ.get('COMET_API_KEY'), \
                                                project_name=args.project_name, \
                                                experiment_name=result_path_stem,\
                                                workspace=args.workspace)

    trainer.logger = tb_logger

    # Load dataset and add dataset specific information to args
    log.info("\nLoading data...")

    train_data, dev_data, test_data = dataset_factory.get_dataset(args)
    train_data.set_args(args)

    train_loader = get_train_dataset_loader(args, train_data, args.batch_size)
    dev_loader = get_eval_dataset_loader(args, dev_data, args.batch_size, True)
    test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)
    eval_train_loader = get_eval_dataset_loader(args, train_data, args.batch_size, False)

    model = lightning.get_lightning_model(args)
    tb_logger.experiment.set_model_graph(model)
    tb_logger.experiment.add_tags(args.comet_tags)

    log.info("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if attr not in ['optimizer_state', 'patient_to_partition_dict', 'path_to_hidden_dict', 'exam_to_year_dict', 'exam_to_device_dict', 'treatment_to_index','drug_to_y']:
            log.info("\t{}={}".format(attr.upper(), value))

    log.info("\n")
    if args.train:
        log.info("-------------\nTrain")
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=dev_loader)
        model_path = trainer.checkpoint_callback.best_model_path
        if args.distributed_backend != 'ddp' or trainer.global_rank == 0:
            log.info("Best model saved to : {}".format(model_path))
            model = model.load_from_checkpoint(model_path, args=args)
        args.model_path = model_path

    log.info("\n")
    if args.print_linear_weights:
        log.info("-------------\nModel Weights:")
        features, weights, odds = get_linear_feature_weights(args, model, train_data)
        for feat, w, o in zip(features, weights, odds):
            print(' - {:<20} {:8.5f}\t{:6.5f}'.format(str(feat) + ':', w, o))

    log.info("\n")
    if args.fine_tune:
        log.info("-------------\nFine tune")
        trainer.max_epochs += args.num_epochs_fine_tune
        model.set_finetune(True)
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=dev_loader)
        model_path = os.path.join(trainer.checkpoint_callback.dirpath, "last.ckpt")
        trainer.save_checkpoint(model_path)
        if args.distributed_backend != 'ddp' or trainer.global_rank == 0:
            log.info("Best model saved to : {}".format(model_path))
            model = model.load_from_checkpoint(model_path, args=args)
        args.model_path = model_path
        model.set_finetune(False)

    log.info("\n")
    if args.dev:
        log.info("-------------\nDev")
        model.save_prefix = 'dev_'
        trainer.test(model, test_dataloaders=dev_loader)

    log.info("\n")
    if args.test:
        log.info("-------------\nTest")
        model.save_prefix = 'test_'
        trainer.test(model, test_dataloaders=test_loader)

    # If timesplit is true, test for timesplit under every test definition
    for task in args.test_tasks:
        args.task = task
        if len(args.test_years) >= 1:
            if args.train_years not in args.test_years:
                args.test_years = [args.train_years]+args.test_years

            # test on each year bucket
            for i in range(len(args.test_years)):
                prefix = 'test_' + args.task + '_' + args.test_years[i] + '_'
                args.current_test_years = args.test_years[i]

                if args.test_onset_buckets:
                    for bucket in args.onset_buckets:
                        args.onset_bucket = bucket
                        model.save_prefix = prefix + 'onset_' + bucket + '_'
                        test_data = dataset_factory.get_specific_dataset(args, 'test')
                        test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)
                        trainer.test(model, test_dataloaders=test_loader)
                
                else: 
                    model.save_prefix = prefix
                    test_data = dataset_factory.get_specific_dataset(args, 'test')
                    test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)
                    trainer.test(model, test_dataloaders=test_loader)
        else:
            model.save_prefix = 'test_' + args.task + '_'
            test_data = dataset_factory.get_specific_dataset(args, 'test')
            test_loader = get_eval_dataset_loader(args, test_data, args.batch_size, False)
            trainer.test(model, test_dataloaders=test_loader)


    if args.eval_train:
        log.info("---\n Now running Eval on train to store final hiddens for each train sample...")
        model.save_prefix = 'eval_train_'
        trainer.test(model, test_dataloaders=eval_train_loader)
    log.info("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open(args.results_path,'wb'))

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parsing.parse_args()
    main(args)
