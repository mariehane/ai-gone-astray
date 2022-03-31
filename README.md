# AI Gone Astray: Source Code
This is the code for "[AI Gone Astray: Technical Supplement](https://arxiv.org/abs/2203.16452)", which investigates the effect of time drift on clinically deployed machine learning models. 

We use MIMIC-IV, a publicly available dataset, to train models that replicate commercial approaches by Dascena and Epic to predict the onset of sepsis, a deadly and yet treatable condition. By doing so, we observe some of these models degrade overtime; most notably an RNN built on Epic features degrades from a 0.729 AUC to a 0.525 AUC over adecade, leading us to investigate technical and clinical drift as root causes of this performance drop.

This repository provides instructions on how to replicate the results of the report, as well as technical details of the training.

If you have any questions, then please contact us via email at
[janicey@mit.edu and ludvig@mit.edu](mailto:janicey@mit.edu,ludvig@mit.edu).

## Setup

The code was originally developed for Python 3.8.5 on Ubuntu 18.04.5 LTS, but should work on any Linux system with at least 64 GB of memory.

### Data Preparation
1. Download [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) and place it in the folder `data`.
2. Extract MIMIC-Code's `mimic_derived.ventilation` table as a csv into `data/ventilation.csv.gz`. Either by downloading it from BigQuery, or by following the [MIMIC-Code instructions](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv) to set up the local database and then exporting the table.

### Installing Dependencies
1. Using Conda, run the following command to create a new environment and install the required dependencies.
      ```
      conda env create -f environment.yml
      ```
2. After this, activate the environment with
      ```
      conda activate ai-gone-astray
      ```
3. Finally [set up Comet by following their instructions.](https://www.comet.ml/docs/quick-start/)

### Sepsis-3
To train and evaluate a single RNN model with Epic features, use one of the following commands:

_Year-Agnostic:_
```
python scripts/main.py  --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn
```

_Year Buckets:_
```
python scripts/main.py  --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 3 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn timesplit
```

A complete list of commands for the remaining feature sets (Dascena, Epic minus ICD Codes), and tasks (Length-of-stay, In-ICU Mortality) can be found in [commands_list.md](commands_list.md).

### Dispatcher

Alternatively, many runs can be spawned at the same time via the use of the dispatcher. For example, to train and evaluate three runs of the RNN model using Epic features under both the Year-Agnostic and Year buckets schemes, run the following command:
```
python scripts/dispatcher.py configs/sepsis_epic_rnn.json configs/sepsis_epic_rnn_timesplit.json --gpus 0 1 2 3 4 5 6 7
```

See the contents of the `configs/` folder to see all the possible inputs to the dispatcher.

## Training Details
We split the data into 60% train, 20% validation and 20% test data. For training, we use the Adam optimizer with a weight decay of 5e-05 and a learning rate of 1e-4 and a momentum of 0.9. The learning rate is divided by 10 when the loss plateaus for 6 epochs. The model is trained for 100 epochs after which the best model is selected according to its validation performance. We train on a single Nvidia Tesla V100 GPU with a mini-batch size of 32.

## Citing
```
@misc{yang2022astray,
      title = {AI Gone Astray: Technical Supplement},
      author = {Janice Yang and Ludvig Karstens and Casey Ross and Adam Yala},
      year = {2022}
      url = {https://arxiv.org/abs/2203.16452},
      doi = {10.48550/ARXIV.2203.16452},
}
```

## License
This code is licensed under the MIT license. Please see [LICENSE](LICENSE) for details.
