# List of commands

## Sepsis-3 

### Dascena Features

_Logistic Regression Model, Year-Agnostic:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_dascena_epic.csv --vent_path data/ventilation.csv.gz --features 'heart rate' 'respiratory rate' temperature age 'systolic blood pressure' 'diastolic blood pressure' 'oxygen saturation' --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags dascena linear --results_path logs/439a12b20f1ee4be2e0ba551bb7ef54d.results
```

_Logistic Regression Model, Year Buckets:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_dascena_epic.csv --vent_path data/ventilation.csv.gz --features 'heart rate' 'respiratory rate' temperature age 'systolic blood pressure' 'diastolic blood pressure' 'oxygen saturation' --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags timesplit dascena linear
```

_RNN, Year-Agnostic_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_dascena_epic.csv --vent_path data/ventilation.csv.gz --features 'heart rate' 'respiratory rate' temperature age 'systolic blood pressure' 'diastolic blood pressure' 'oxygen saturation' --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags sepsis dascena rnn
```

_RNN, Year Buckets_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_dascena_epic.csv --vent_path data/ventilation.csv.gz --features 'heart rate' 'respiratory rate' temperature age 'systolic blood pressure' 'diastolic blood pressure' 'oxygen saturation' --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags sepsis dascena timesplit rnn
```

### Epic Features

_Logistic Regression Model, Year-Agnostic:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags epic linear
```

_Logistic Regression Model, Year-Buckets:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags timesplit epic linear
```

_RNN, Year-Agnostic_:
```
python scripts/main.py  --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn
```

_RNN, Year Buckets_:
```
python scripts/main.py  --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 3 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn timesplit
```

### Epic Features Minus ICD Codes
_Logistic Regression Model, Year-Agnostic:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --feature_remove hiv obesity 'coronary artery disease' 'congestive heart failure' 'chronic obstructive pulmonary disease (COPD)' 'chronic kidney disease' 'chronic liver disease' diabetes hypertension --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags epic linear icd-remove
```

_Logistic Regression Model, Year-Buckets:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --feature_remove hiv obesity 'coronary artery disease' 'congestive heart failure' 'chronic obstructive pulmonary disease (COPD)' 'chronic kidney disease' 'chronic liver disease' diabetes hypertension --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --flatten_timeseries --model_name linear --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags epic linear timesplit icd-remove
```

_RNN, Year-Agnostic_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --feature_remove hiv obesity 'coronary artery disease' 'congestive heart failure' 'chronic obstructive pulmonary disease (COPD)' 'chronic kidney disease' 'chronic liver disease' diabetes hypertension --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn icd-remove
```

_RNN, Year Buckets_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-sepsis --dataset_path data/mimiciv --item_map_path data/item_map_epic.csv --vent_path data/ventilation.csv.gz --feature_remove hiv obesity 'coronary artery disease' 'congestive heart failure' 'chronic obstructive pulmonary disease (COPD)' 'chronic kidney disease' 'chronic liver disease' diabetes hypertension --sepsis_consider_sofa_difference --dascena_control --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 30 --gap_hours 6 --min_icu_stay 24 --max_icu_stay 10 --cache_dir cache/ --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --rnn_pool_name Simple_AttentionPool --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --drop_last --comet_tags epic rnn timesplit icd-remove
```

## Length-of-stay >= 3 days

_RNN, Year-Agnostic_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-los --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task los --los_threshold 3 --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags los rnn
```

_RNN, Year Buckets_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-los --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task los --los_threshold 3 --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags los timesplit rnn 
```

_Logistic Regression Model, Year-Agnostic:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-los --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task los --los_threshold 3 --flatten_timeseries --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --split_probs 0.6 0.2 0.2 --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags los linear
```

_Logistic Regression Model, Year-Buckets:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-los --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task los --los_threshold 3 --flatten_timeseries --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags los timesplit linear
```

## In-ICU Mortality

_RNN, Year-Agnostic_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-icumort --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task icumort --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags icumort rnn
```

_RNN, Year Buckets_:
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-icumort --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task icumort --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name rnn --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags icumort timesplit rnn
```

_Logistic Regression Model, Year-Agnostic:_
```
python scripts/main.py  --batch_size 32 --gpus 1 --dataset mimic-iv-icumort --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task icumort --flatten_timeseries --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --split_probs 0.6 0.2 0.2 --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --rnn_num_layers 2 --rnn_hidden_dim 128 --rnn_dropout 0.1 --rnn_bidirectional --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags icumort linear
```

_Logistic Regression Model, Year-Buckets:_
```
python scripts/main.py --batch_size 32 --gpus 1 --dataset mimic-iv-icumort --dataset_path data/mimiciv --item_map_path data/item_map_chartevents.csv --cache_dir cache/ --task icumort --flatten_timeseries --impute_method simple --cross_val_seed 1 --min_patient_age 15 --data_hours 24 --min_hours 24 --gap_hours 0 --min_icu_stay 36 --max_icu_stay 10 --timesplit --train_years 2008-2010 --test_years 2011-2013 2014-2016 2017-2019 --split_probs 0.6 0.2 0.2 --model_name linear --num_layers 4 --hidden_dim 64 --dropout 0.1 --weight_decay 5e-05 --momentum 0.9 --max_epochs 100 --init_lr 0.0001 --num_workers 4 --optimizer adam --patience 6 --save_dir snapshot/ --train --eval_train --dev --test --comet_tags icumort timesplit linear
```
