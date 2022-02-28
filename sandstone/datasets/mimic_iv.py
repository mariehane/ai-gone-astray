import time
from pathlib import Path
from typing import Tuple, Sequence
from collections import Counter
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm

from sandstone.datasets.factory import RegisterDataset
from sandstone.utils.generic import log, md5

import warnings
warnings.simplefilter("ignore")

class MIMIC_IV_Abstract_Dataset(data.Dataset):
    """Abstract class for different MIMIC-IV tasks.
    Handles data loading, caching, splitting, and various generic preprocessing steps.
    """

    def __init__(self, args, split_group):
        super(MIMIC_IV_Abstract_Dataset, self).__init__()

        self.args = args
        self.split_group = split_group

        cache_static_filename = get_cache_filename('static', args=args)
        cache_hourly_filename = get_cache_filename('hourly', args=args)

        print(f"Loading item mapping ({args.item_map_path})")
        item_mapping = pd.read_csv(args.item_map_path, low_memory=False)

        if Path(args.cache_dir, cache_static_filename).is_file() and Path(args.cache_dir, cache_hourly_filename).is_file():
            print("Loading cached static_df and aggregated_df:", cache_static_filename, cache_hourly_filename)
            static_df = pd.read_parquet(Path(args.cache_dir, cache_static_filename))
            aggregated_df = pd.read_parquet(Path(args.cache_dir, cache_hourly_filename))
        else:
            # compute which csvs are needed
            task_csv_subset = set(self.task_specific_features.keys())
            features_csv_subset = set(item_mapping.origin.loc[item_mapping.origin != 'static'].dropna())
            # by default, patients, chartevents, admissions and icustays are loaded
            self.csv_subset = set(('patients', 'chartevents', 'admissions', 'icustays')).union(task_csv_subset).union(features_csv_subset)

            raw_dataframes = load_data(args.dataset_path, subset=self.csv_subset, nrows=args.nrows, chunksize=args.chunksize, cache_dir=args.cache_dir)
            static_df, aggregated_df = self.create_dataframes(args, item_mapping, **raw_dataframes)

            # cache final dataframes
            static_df.to_parquet(Path(args.cache_dir, cache_static_filename))
            aggregated_df.to_parquet(Path(args.cache_dir, cache_hourly_filename))

        print("Generating labels")
        self.create_labels(static_df, aggregated_df, task=args.task, threshold=args.los_threshold)

        if args.dataset == 'mimic-iv-sepsis':
            print(f"Extracting {args.data_hours} hours of data") 
            aggregated_df = self.extract_timerange(args, aggregated_df, task=args.task)
            
            print("Adding onset hour to static_df")
            onset = aggregated_df.groupby('hadm_id')[args.task+'_onset_hour'].mean()
            static_df = static_df.merge(onset, how='left', on='hadm_id')
            # filter static_df to only include patients in aggregated_df
            static_df = static_df[static_df.hadm_id.isin(aggregated_df.hadm_id.unique())]
        

        print("Filter for just feature columns")
        static_cols = ['subject_id', 'hadm_id', 'intime', 'y', args.task+'_onset_hour']
        cols_to_keep = ['hadm_id', 'hour']
        if len(args.features) != 0:
            # convert to lower case
            args.features = [x.lower() for x in args.features]
            if args.group_by_level2:
                static_cols.extend(args.features)
                cols_to_keep.extend(args.features)
            else:
                feature_ids = list(item_mapping.loc[item_mapping['LEVEL2'].str.lower().isin(args.features)]['itemid'].map(str))
                static_cols.extend(feature_ids)
                cols_to_keep.extend(feature_ids)
        else:
            static_cols.extend(list(item_mapping.itemid.map(str)))
            if args.group_by_level2:
                cols_to_keep.extend(list(item_mapping.LEVEL2))
            else:
                cols_to_keep.extend(list(item_mapping.itemid.map(str)))

        if args.feature_search is not None:
            args.feature_search = args.feature_search.lower()
            if args.group_by_level2:
                print("Search feature:", args.feature_search)
                static_cols.extend(args.feature_search)
                cols_to_keep.extend(args.feature_search)
            else:
                search_ids = list(item_mapping.loc[item_mapping['LEVEL2'].str.lower() == (args.feature_search)]['itemid'].map(str))
                print("Search IDs:", search_ids)
                cols_to_keep.extend(search_ids)
                static_cols.extend(search_ids)

        if len(args.feature_remove) != 0:
            # convert to lower case
            args.feature_remove = [x.lower() for x in args.feature_remove]

            if args.group_by_level2:
                remove_ids = args.feature_remove
            else:
                remove_ids = list(item_mapping.loc[item_mapping['LEVEL2'].str.lower().isin(args.feature_remove)]['itemid'].map(str))

            for feature in remove_ids:
                if feature in cols_to_keep:
                    print("Removed feature:", feature)
                    cols_to_keep.remove(feature)
                if feature in static_cols: 
                    static_cols.remove(feature)


        original_cols = [c for c in cols_to_keep if c in aggregated_df.columns]
        if args.impute_method == 'simple':
            exist_cols = [c+'_exist' for c in original_cols if c not in ['hadm_id', 'hour']]
            time_cols = [c+'_time_since' for c in original_cols if c not in ['hadm_id', 'hour']]
            
            cols_to_keep.extend(exist_cols)
            cols_to_keep.extend(time_cols)

        static_df = static_df.loc[:, static_df.columns.isin(static_cols)]
        aggregated_df = aggregated_df.loc[:,aggregated_df.columns.isin(cols_to_keep)]

        if args.dataset == 'mimic-iv-sepsis':
            print(f"Re-indexing and zero filling")
            aggregated_df = reindex_timeseries(aggregated_df)
            aggregated_df.fillna({x:0 for x in original_cols}, inplace=True)

            if args.impute_method == 'simple': 
                aggregated_df.fillna({x:0 for x in exist_cols}, inplace=True)
                aggregated_df.fillna({x:100 for x in time_cols}, inplace=True)

        print("Static df size:", static_df.shape)
        print("Static df columns:", static_df.columns)
        print("Aggregated df size:", aggregated_df.shape)
        print("Aggregated df columns:", aggregated_df.columns)
        print("Static df stats:")
        print(static_df.describe())
        print("Aggregated df stats:")
        print(aggregated_df.describe())

        print("Binarize/One-hot encode categorical feature columns")
        if 'gender' in static_df.columns:
            static_df['gender'] = (static_df.gender == 'M').astype(bool)

        for col in ['marital_status', 'ethnicity']:
            if col in static_df.columns:
                dummies = pd.get_dummies(static_df[col]).add_prefix(col+"_").astype(bool)
                static_df.drop(columns=col, inplace=True)
                static_df[dummies.columns] = dummies
    

        self.assign_splits(static_df)

        if args.normalize is not None:
            print("Normalizing values to zero-mean and unit variance.")
            if args.group_by_level2:
                normalize_feats = set(args.normalize)
            else:
                normalize_feats = set(item_mapping.loc[item_mapping['LEVEL2'].isin(args.normalize)].itemid.unique())

            static_norm_cols = list(normalize_feats.intersection(static_df.columns))
            hourly_norm_cols = list(normalize_feats.intersection(aggregated_df.columns))

            unused_norm_cols = normalize_feats.difference(set(static_norm_cols + hourly_norm_cols))
            if len(unused_norm_cols) != 0:
                print("WARNING: Couldn't find specified columns to normalize by: {}!".format(unused_norm_cols))

            static_train = static_df.loc[static_df.split_group == 'train']
            static_normalize_df = static_train[static_norm_cols]
            hourly_normalize_df = aggregated_df.loc[aggregated_df.hadm_id.isin(static_train.hadm_id.unique()), hourly_norm_cols]

            # compute stats over train data
            static_mean, static_std = static_normalize_df.mean(), static_normalize_df.std()
            hourly_mean, hourly_std = hourly_normalize_df.mean(), hourly_normalize_df.std()

            # prevent division by zero
            static_std.loc[static_std == 0] = 1
            hourly_std.loc[hourly_std == 0] = 1

            # apply to whole dataset
            static_df[static_norm_cols] = (static_df[static_norm_cols] - static_mean) / static_std
            aggregated_df[hourly_norm_cols] = (aggregated_df[hourly_norm_cols] - hourly_mean) / hourly_std
        
        if args.flatten_timeseries:
            flattened_df = flatten_timeseries(aggregated_df)
            static_df = static_df.merge(flattened_df, on='hadm_id')
        elif args.timeseries_moments:
            moments_df = compute_timeseries_moments(aggregated_df, args.timeseries_moments)
            static_df = static_df.merge(moments_df, on='hadm_id')

        static_df.columns = static_df.columns.map(str)
        self.static_features = [col for col in static_df.columns if col not in ['y', 'subject_id', 'hadm_id', 'intime', 'split_group', args.task+'_onset_hour']]
        self.timeseries_features = [col for col in aggregated_df.columns if col not in ['hadm_id', 'charttime', 'hour']]

        static_df = static_df.loc[static_df['split_group'] == split_group]
        if not args.static_only:
            # if non-flattened hourly data is used, also filter aggregated_df
            aggregated_df = aggregated_df.loc[aggregated_df['hadm_id'].isin(static_df['hadm_id'].unique())]
        static_df.drop(columns='split_group', inplace=True)
        
        if args.static_only:
            self.dataset = self.create_dataset(static_df)
        else:
            self.dataset = self.create_dataset(static_df, aggregated_df)

        # Class weighting
        label_dist = [d['y'] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        self.weights = [ label_weights[d['y']] for d in self.dataset]

        log(self.get_summary_statement(self.args.task, split_group, self.args.current_test_years, self.args.onset_bucket, label_counts), args)

    @property
    def task(self):
        raise NotImplementedError("Abstract method needs to be overridden!")

    @property
    def task_specific_features(self, task=None):
        """Defines some itemids/gsns/icd_codes that are needed for the task.

        Returns:
            a dictionary mapping origin dataset -> list of itemids.
        """
        return {}

    def create_dataframes(self, args, item_mapping, **raw_dataframes):
        """Preprocesses raw dataframes into static_df and aggregated_df.

        Returns:
            - static_df 
                - must include columns 'hadm_id', and 'y' for the label.
                - any additional columns will be used as input features for prediction.
            - timeseries_df
        """
        raise NotImplementedError("Abstract method needs to be overridden!")

    def assign_splits(self, meta):
        if self.args.timesplit: 
            # assign train_years as a list of years [2008, 2010] inclusive for instance. 
            train_start, train_end = map(int, self.args.train_years.split('-'))
            meta['split_group'] = None

            meta.loc[(meta['intime'].dt.year>=train_start) & (meta['intime'].dt.year<=train_end), 'split_group'] = 'train'

            # dev will be a subset of train, of proportion split_probs[dev]
            dev_prob = self.args.split_probs[1]
            train_rows = meta[meta.split_group=='train'].shape[0]
            dev_rows = int(dev_prob*train_rows)
            meta.loc[meta[meta['split_group']=='train'].head(dev_rows).index, 'split_group'] = 'dev'

            # if testing on training years, then final split is test set 
            if self.args.train_years == self.args.current_test_years:
                test_prob = self.args.split_probs[2]
                test_rows = int(test_prob*train_rows)
                mask = meta.index.isin(meta[meta['split_group']=='train'].tail(test_rows).index)
        
            else:
                test_start, test_end = map(int, self.args.current_test_years.split('-'))
                mask = meta['intime'].dt.year>=test_start
                mask &= meta['intime'].dt.year<=test_end

            # adding to the mask onset bucket
            if self.args.onset_bucket is not None: 
                hour_start, hour_end = map(int, self.args.onset_bucket.split('-'))
                mask &= meta[self.args.task+'_onset_hour'] >= hour_start
                mask &= meta[self.args.task+'_onset_hour'] <= hour_end
            
            meta.loc[mask, 'split_group'] = 'test'
        
        else:
            subject_ids = list(sorted(meta['subject_id'].unique()))
            start_idx = 0
            meta['split_group'] = None
            for split, prob in zip(['train', 'dev', 'test'], self.args.split_probs):
                end_idx = start_idx + int(len(subject_ids) * prob)
                start = subject_ids[start_idx]
                end = subject_ids[end_idx-1]
                meta.loc[(meta['subject_id'] >= start) & (meta['subject_id'] <= end), 'split_group'] = split
                start_idx = end_idx
            if meta.loc[meta['subject_id']==subject_ids[end_idx-1]]['split_group'].isnull().any():
                meta.loc[meta['subject_id']==subject_ids[end_idx-1], 'split_group'] = split
        return meta

    def create_dataset(self, static_df, aggregated_df=None):
        """Turns DataFrames into a list of samples, which are dicts containing 'pid', 'x', 'y', and 
        possibly 'x_timeseries' keys
        """
        dataset = []
        pids = static_df['subject_id'].values.astype(np.int32)
        hadm_ids = static_df['hadm_id'].values.astype(np.int32)
        ys = static_df['y'].values.astype(np.float32)
        xs = static_df[self.static_features].values.astype(np.float32)

        for i in tqdm(range(len(pids)), desc='Creating dataset', total=len(pids)):
            patient_dict = {}
            patient_dict['pid'] = pids[i]
            patient_dict['y'] = ys[i]
            patient_dict['x'] = xs[i]

            if aggregated_df is not None:
                patient_rows = aggregated_df.loc[aggregated_df.hadm_id == hadm_ids[i]]
                assert len(patient_rows) > 0, "Found patient with no timeseries data!"
                x_timeseries = patient_rows[self.timeseries_features].values.astype(np.float32)
                patient_dict['x_timeseries'] = x_timeseries

            dataset.append(patient_dict)

        return dataset

    def create_labels(self, static_df, aggregated_df, task, threshold):
        """Generates per-patient labels for the given task
        Returns:
            - static_df with an extra 'y' column
        """
        raise NotImplementedError("Abstract method needs to be overridden!")

    def extract_timerange(self, args, aggregated_df, task):
        """Extracts a fixed no. of hours of data to predict from
        """
        raise NotImplementedError("Abstract method needs to be overridden!")
    
    def get_summary_statement(self, task, split_group, years, hours, class_balance):
        return "Created MIMIC-IV {} {} dataset for years {} and onset hours {} with the following class balance:\n{}".format(task, split_group, years, hours, class_balance)

    def set_args(self, args):
        args.num_classes = 2

        args.input_dim = len(self.static_features)
        if not args.flatten_timeseries:
            args.timeseries_dim = len(self.timeseries_features)
            args.timeseries_len = args.data_hours

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


@RegisterDataset("mimic-iv-sepsis")
class MIMIC_IV_Sepsis_Dataset(MIMIC_IV_Abstract_Dataset):

    @property
    def task(self):
        return "Sepsis-3"

    @property
    def task_specific_features(self):
        return {
            'inputevents': [221662, 221653, 221289, 221906], # dopamine, dobutamine, epinephrine, norepinephrine
            'outputevents': [226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557,
                            226558, 227488, 227489], # for urine output
            'labevents': [51265, 50885, 50912, 50821, 51301], # platelets, bilirubin, creatinine, PO2, WBC-count
            'chartevents': [223835, 220739, 223900, 223901, 223849, 229314, # FiO2, GCS-Eye, GCS-Verbal, GCS-Motor, vent_mode, vent_mode (Hamilton)
                            223762, 223761, 220045, 220210, 224690], # temp_C, temp_F, heart rate, resp rate, resp rate (total)
            'microbiologyevents': None, # all microbio samples (no id filtering happens on microbioevents, so None can be used here)
            'prescriptions': None,
        }
    
    def create_dataframes(self, args, item_mapping, patients, chartevents, admissions, icustays,
                          inputevents, labevents, microbiologyevents=None, prescriptions=None, outputevents=None,
                          diagnoses_icd=None, procedureevents=None, **extra_dfs):
        # filter patients and merge data (code from before)
        admissions, patients, icustays = filter_eligible_patients(admissions, patients, icustays, 
                                                                  args.min_patient_age, args.min_hours, args.gap_hours,
                                                                  args.min_icu_stay, args.max_icu_stay)
        chartevents = filter_table_patients(chartevents, patients)
        labevents = filter_table_patients(labevents, patients)
        inputevents = filter_table_patients(inputevents, patients)
        microbiologyevents = filter_table_patients(microbiologyevents, patients)
        prescriptions = filter_table_patients(prescriptions, patients)
        outputevents = filter_table_patients(outputevents, patients)
        diagnoses_icd = filter_table_patients(diagnoses_icd, patients)
        procedureevents = filter_table_patients(procedureevents, patients)

        print("Merging static data...")
        static_df = patients[["subject_id", "gender", "anchor_age"]]
        static_df = static_df.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime", "insurance", "admission_type", "marital_status", "ethnicity"]],
                                    how="inner", on="subject_id")
        static_df = static_df.merge(icustays[["hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"]],
                                    how="inner", on="hadm_id")
        static_df.rename(columns={"anchor_age": "age", "stay_id": "icustay_id"}, inplace=True)
        
        print("Filter events")
        chartevents_features = item_mapping.loc[item_mapping.origin == 'chartevents'].itemid.astype(int).tolist()
        inputevents_features = item_mapping.loc[item_mapping.origin == 'inputevents'].itemid.astype(int).tolist()
        outputevents_features = item_mapping.loc[item_mapping.origin == 'outputevents'].itemid.astype(int).tolist()
        labevents_features = item_mapping.loc[item_mapping.origin == 'labevents'].itemid.astype(int).tolist()
        procedureevents_features = item_mapping.loc[item_mapping.origin == 'procedureevents'].itemid.astype(int).tolist()
        prescriptions_features = item_mapping.loc[item_mapping.origin == 'prescriptions'].itemid.tolist()

        inputevents_features.extend(self.task_specific_features['inputevents'])
        outputevents_features.extend(self.task_specific_features['outputevents'])
        labevents_features.extend(self.task_specific_features['labevents'])
        chartevents_features.extend(self.task_specific_features['chartevents'])

        filtered_inputevents = inputevents.loc[inputevents['itemid'].isin(inputevents_features)]
        filtered_outputevents = outputevents.loc[outputevents['itemid'].isin(outputevents_features)]
        filtered_labevents = labevents.loc[labevents['itemid'].isin(labevents_features)]
        filtered_chartevents = filter_variables(chartevents, chartevents_features)
        filtered_prescriptions = prescriptions.loc[prescriptions['gsn'].isin(prescriptions_features)]
        antibiotics = filter_antibiotics(prescriptions)
        filtered_diagnoses = filter_diagnoses(diagnoses_icd, item_mapping)
        filtered_procedures = procedureevents.loc[procedureevents['itemid'].isin(procedureevents_features)]

        # standardize units
        print("Standardizing units")
        filtered_chartevents = standardize_units(filtered_chartevents, item_mapping)

        # merge diagnoses with static_df
        filtered_diagnoses['value'] = 1
        pivot_diagnoses = filtered_diagnoses.pivot_table(index='hadm_id', columns='icd_code', values ='value')
        static_df = static_df.merge(pivot_diagnoses, on='hadm_id', how='left')
        static_df[pivot_diagnoses.columns] = static_df[pivot_diagnoses.columns].fillna(0)

        print("Filter events to stay")
        filtered_inputevents.rename(columns={"starttime": "charttime"}, inplace=True)
        antibiotics.rename(columns={'starttime':'charttime'}, inplace=True)
        filtered_procedures.rename(columns={'starttime':'charttime'}, inplace=True)
        chartlab_events = pd.concat([filtered_chartevents, filtered_labevents], join='outer')
        filtered_prescriptions.rename(columns={'starttime':'charttime', 'gsn':'itemid'}, inplace=True)
        # Pass chartevents dataframe and inputevents through hourly aggregation
        chartlab_events = filter_events_to_stay(chartlab_events, static_df)
        filtered_inputevents = filter_events_to_stay(filtered_inputevents, static_df)
        microbiologyevents = filter_events_to_stay(microbiologyevents, static_df)
        antibiotics = filter_events_to_stay(antibiotics, static_df)
        filtered_prescriptions = filter_events_to_stay(filtered_prescriptions, static_df)
        filtered_outputevents = filter_events_to_stay(filtered_outputevents, static_df)
        filtered_procedures = filter_events_to_stay(filtered_procedures, static_df)

        if args.group_by_level2:
            print("Group itemids by actual feature they represent")
            item_mapping_chartlab = item_mapping.loc[item_mapping.origin == 'chartevents', ['itemid', 'LEVEL2']].astype({'itemid': int})
            chartlab_events = chartlab_events.merge(item_mapping_chartlab, on='itemid', how='left')
            group_mask = ~chartlab_events.LEVEL2.isna()
            chartlab_events.loc[group_mask, 'itemid'] = chartlab_events.loc[group_mask, 'LEVEL2']

        print("Hourly aggregation")
        # fill NaN with 1 for incisions etc. 
        chartlab_events[['value','valuenum']].fillna(1, inplace=True)
        aggregated_df = hourly_aggregation(chartlab_events, static_df, filtered_inputevents, antibiotics, microbiologyevents, filtered_outputevents, filtered_procedures, filtered_prescriptions)

        print("Calculate SOFA, SI and Sepsis-3")
        # import vents -- can move this code into SOFA score if necessary
        vents_df = pd.read_csv(args.vent_path, low_memory=False)
        vents_df = pd.merge(vents_df, static_df[['subject_id', 'hadm_id', 'icustay_id']],
                            how='inner', left_on='stay_id', right_on='icustay_id') # filter for relevant stay & patients
        vents_df['starttime'] = pd.to_datetime(vents_df.starttime)
        vents_df['endtime'] = pd.to_datetime(vents_df.endtime)
        vents_df = anchor_dates(vents_df, ['starttime', 'endtime'], patients)
        aggregated_df = add_vents(aggregated_df, vents_df)

        # Calculate SOFA scores as additional columns
        aggregated_df = calculate_SOFA(aggregated_df)

        # Calculate Suspicion of Infection as an additional column
        aggregated_df = calculate_SI(aggregated_df)

        # Calculate Sepsis from SOFA and SI
        aggregated_df = calculate_sepsis(aggregated_df, task="sepsis3", consider_difference=args.sepsis_consider_sofa_difference, decrease_baseline=args.sepsis_decrease_sofa_baseline)

        # Add SIRS definition as column
        # XXX: commented out because of conflict with itemid grouping
        #aggregated_df = calculate_SIRS(aggregated_df)

        # Calculate Sepsis from SIRS and SI
        #aggregated_df = calculate_sepsis(aggregated_df, task="sepsis1", consider_difference=args.sepsis_consider_sofa_difference, decrease_baseline=args.sepsis_decrease_sofa_baseline)

        # print("Filtering out patients without enough data")
        # # Filtering out patients without enough data: 
        # counts = aggregated_df['hadm_id'].value_counts()
        # aggregated_df = aggregated_df[aggregated_df['hadm_id'].isin(counts[counts>(args.data_hours+args.gap_hours)].index)]

        print("Computing approximate real dates...")
        static_df = anchor_dates(static_df, ["admittime", "dischtime", "intime", "outtime"], patients)
        if 'charttime' in aggregated_df.columns:
            aggregated_df = aggregated_df.merge(static_df[['hadm_id','subject_id']], on='hadm_id')
            aggregated_df = anchor_dates(aggregated_df, ['charttime'], patients)

        # drop patients where any one feature has no vitals
        if args.dascena_drop: 
            print("Dropping patients with any vital missing")
            categories = ["heart rate", "respiratory rate", "temperature", "systolic blood pressure",
              "diastolic blood pressure", "oxygen saturation"]
            for vital in categories:
                if args.group_by_level2:
                    if vital not in aggregated_df.columns:
                        continue
                    mask = aggregated_df.set_index("hadm_id")[vital].notnull().groupby(level=0).any()
                else:
                    ids = list(item_mapping.loc[item_mapping['LEVEL2'].str.lower() == vital]['itemid'].map(str))
                    valid_ids = [i for i in ids if i in aggregated_df.columns]
                    if len(valid_ids) == 0:
                        continue
                    mask = aggregated_df.set_index("hadm_id")[valid_ids].notnull().groupby(level=0).any().any(axis=1)
                aggregated_df = aggregated_df.set_index("hadm_id")[mask].reset_index()


        # Impute
        print("Imputing NaNs")
        
        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])
        print("- Ratio of Nans:", aggregated_df.isna().sum().sum() / total_values)
        
        ignore_cols = ['hadm_id', 'charttime', 'hour', 'subject_id'] + list(aggregated_df.select_dtypes(include="bool").columns)
        impute_cols = [col for col in aggregated_df.columns if col not in ignore_cols]
        aggregated_df = impute_timeseries(aggregated_df, method=args.impute_method, feature_cols=impute_cols)

        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])

        print("After imputation:")
        print("- Ratio of zeroes:", (aggregated_df == 0).sum().sum() / total_values)

        return static_df, aggregated_df

    def create_labels(self, static_df, aggregated_df, task='sepsis3', threshold=None):
        # generate per-patient sepsis3 label
        sepsis_hadm_ids = aggregated_df.hadm_id[aggregated_df[task] == True].unique()
        static_df['y'] = False
        static_df.loc[static_df.hadm_id.isin(sepsis_hadm_ids), 'y'] = True
        
    def extract_timerange(self, args, aggregated_df, task='sepsis3'):
        sepsis_onset_hour = aggregated_df[aggregated_df[task+'_onset']][['hadm_id', 'hour']]
        sepsis_onset_hour.rename(columns={'hour': task+'_onset_hour'}, inplace=True)

        aggregated_df = extract_data_prior_to_event(aggregated_df, sepsis_onset_hour, key='hadm_id', events_hour_column=task+'_onset_hour', 
                                                    gap_hours=args.gap_hours, data_hours=args.data_hours, case_control=args.case_control, dascena_control=args.dascena_control)

        return aggregated_df



@RegisterDataset("mimic-iv-los")
class MIMIC_IV_Los_Dataset(MIMIC_IV_Abstract_Dataset):

    @property
    def task(self):
        return "Length of Stay"
    
    def create_dataframes(self, args, item_mapping, patients, chartevents, admissions, icustays):
        admissions, patients, icustays = filter_eligible_patients(admissions, patients, icustays, 
                                                                  args.min_patient_age, args.min_hours, args.gap_hours,
                                                                  args.min_icu_stay, args.max_icu_stay)
        chartevents = filter_table_patients(chartevents, patients)
        print("Merging static data...")
        static_df = patients[["subject_id", "gender", "anchor_age"]]
        static_df = static_df.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime", "insurance", "admission_type", "marital_status", "ethnicity"]],
                                    how="inner", on="subject_id")
        static_df = static_df.merge(icustays[["hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"]],
                                    how="inner", on="hadm_id")
        static_df.rename(columns={"anchor_age": "age", "stay_id": "icustay_id"}, inplace=True)

        print("Filter events")
        chartevents_features = item_mapping.loc[item_mapping.origin == 'chartevents'].itemid.astype(int).tolist()
        filtered_chartevents = filter_variables(chartevents, chartevents_features)

        print("Standardizing units")
        filtered_chartevents = standardize_units(filtered_chartevents, item_mapping)

        print("Filter events to stay")
        filtered_chartevents = filter_events_to_stay(filtered_chartevents, static_df)

        if args.group_by_level2:
            print("Group itemids by actual feature they represent")
            item_mapping_chart = item_mapping.loc[item_mapping.origin == 'chartevents', ['itemid', 'LEVEL2']].astype({'itemid': int})
            filtered_chartevents = filtered_chartevents.merge(item_mapping_chart, on='itemid', how='left')
            group_mask = ~filtered_chartevents.LEVEL2.isna()
            filtered_chartevents.loc[group_mask, 'itemid'] = filtered_chartevents.loc[group_mask, 'LEVEL2']

        print("Hourly aggregation")
        aggregated_df = hourly_aggregation(filtered_chartevents, static_df)

        print("Computing approximate real dates...")
        static_df = anchor_dates(static_df, ["admittime", "dischtime", "intime", "outtime"], patients)
        if 'charttime' in aggregated_df.columns:
            aggregated_df = aggregated_df.merge(static_df[['hadm_id','subject_id']], on='hadm_id')
            aggregated_df = anchor_dates(aggregated_df, ['charttime'], patients)

        print(f"Extracting {args.data_hours} hours of data") 
        aggregated_df = self.extract_timerange(args, aggregated_df, task=args.task)

        print("Reindexing timeseries")
        aggregated_df = reindex_timeseries(aggregated_df)

        # Imputing 
        print("Imputing NaNs")
        
        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])
        print("- Ratio of Nans:", aggregated_df.isna().sum().sum() / total_values)

        impute_cols = [col for col in aggregated_df.columns if col not in ['hadm_id', 'charttime', 'hour', 'subject_id']]
        aggregated_df = impute_timeseries(aggregated_df, method=args.impute_method, feature_cols=impute_cols)

        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])

        print("After imputation:")
        print("- Ratio of zeroes:", (aggregated_df == 0).sum().sum() / total_values)

        
        # filter static_df to only include patients in aggregated_df
        static_df = static_df[static_df.hadm_id.isin(aggregated_df.hadm_id.unique())]


        return static_df, aggregated_df

    def create_labels(self, static_df, aggregated_df, task=None, threshold=4):
        static_df['y'] = static_df['los'] >= threshold

    # extract first data_hours data from each patient
    def extract_timerange(self, args, aggregated_df, task=None):
        # aggregated_df['hour'] = aggregated_df.groupby('hadm_id')['hour'].rank('first')
        df = aggregated_df.loc[aggregated_df['hour']<= args.data_hours]
        return df
        
@RegisterDataset("mimic-iv-icumort")
class MIMIC_IV_ICUMort_Dataset(MIMIC_IV_Abstract_Dataset):

    @property
    def task(self):
        return "ICU Mortality"
    
    def create_dataframes(self, args, item_mapping, patients, chartevents, admissions, icustays):
        admissions, patients, icustays = filter_eligible_patients(admissions, patients, icustays, 
                                                                  args.min_patient_age, args.min_hours, args.gap_hours,
                                                                  args.min_icu_stay, args.max_icu_stay)
        chartevents = filter_table_patients(chartevents, patients)
        print("Merging static data...")
        static_df = patients[["subject_id", "gender", "anchor_age"]]
        static_df = static_df.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "insurance", "admission_type", "marital_status", "ethnicity"]],
                                    how="inner", on="subject_id")
        static_df = static_df.merge(icustays[["hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"]],
                                    how="inner", on="hadm_id")
        static_df['death_in_icu'] = (~static_df['deathtime'].isna()) & (static_df['deathtime'] >= static_df['intime']) & \
                                            (static_df['deathtime'] <= static_df['outtime'])
        static_df.rename(columns={"anchor_age": "age", "stay_id": "icustay_id"}, inplace=True)

        print("Filter events")
        chartevents_features = item_mapping.loc[item_mapping.origin == 'chartevents'].itemid.astype(int).tolist()
        filtered_chartevents = filter_variables(chartevents, chartevents_features)

        print("Standardizing units")
        filtered_chartevents = standardize_units(filtered_chartevents, item_mapping)

        print("Filter events to stay")
        filtered_chartevents = filter_events_to_stay(filtered_chartevents, static_df)

        if args.group_by_level2:
            print("Group itemids by actual feature they represent")
            item_mapping_chart = item_mapping.loc[item_mapping.origin == 'chartevents', ['itemid', 'LEVEL2']].astype({'itemid': int})
            filtered_chartevents = filtered_chartevents.merge(item_mapping_chart, on='itemid', how='left')
            group_mask = ~filtered_chartevents.LEVEL2.isna()
            filtered_chartevents.loc[group_mask, 'itemid'] = filtered_chartevents.loc[group_mask, 'LEVEL2']

        print("Hourly aggregation")
        aggregated_df = hourly_aggregation(filtered_chartevents, static_df)

        print("Computing approximate real dates...")
        static_df = anchor_dates(static_df, ["admittime", "dischtime", "intime", "outtime"], patients)
        if 'charttime' in aggregated_df.columns:
            aggregated_df = aggregated_df.merge(static_df[['hadm_id','subject_id']], on='hadm_id')
            aggregated_df = anchor_dates(aggregated_df, ['charttime'], patients)

        print(f"Extracting {args.data_hours} hours of data") 
        aggregated_df = self.extract_timerange(args, aggregated_df, task=args.task)

        print("Reindexing timeseries")
        aggregated_df = reindex_timeseries(aggregated_df)

        # Imputing 
        print("Imputing NaNs")
        
        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])
        print("- Ratio of Nans:", aggregated_df.isna().sum().sum() / total_values)

        impute_cols = [col for col in aggregated_df.columns if col not in ['hadm_id', 'charttime', 'hour', 'subject_id']]
        aggregated_df = impute_timeseries(aggregated_df, method=args.impute_method, feature_cols=impute_cols)

        total_values = (aggregated_df.shape[0] * aggregated_df.shape[1])

        print("After imputation:")
        print("- Ratio of zeroes:", (aggregated_df == 0).sum().sum() / total_values)

        
        # filter static_df to only include patients in aggregated_df
        static_df = static_df[static_df.hadm_id.isin(aggregated_df.hadm_id.unique())]

        return static_df, aggregated_df

    def create_labels(self, static_df, aggregated_df, task=None, threshold=None):
        static_df['y'] = static_df['death_in_icu']

    # extract first data_hours data from each patient
    def extract_timerange(self, args, aggregated_df, task=None):
        # aggregated_df['hour'] = aggregated_df.groupby('hadm_id')['hour'].rank('first')
        df = aggregated_df.loc[aggregated_df['hour']<= args.data_hours]
        return df

# args that affect cache
CACHE_ARGS = ['dataset', 'min_patient_age', 'data_hours', 'min_hours', 'gap_hours', 'min_icu_stay', 'max_icu_stay', 'item_map_path',
              'sepsis_consider_sofa_difference', 'sepsis_decrease_sofa_baseline', 'group_by_level2', 'impute_method', 'dascena_drop']

def get_cache_filename(filename, args, extension='parquet'):
    args_dict = vars(args)
    args_str = ""
    for arg in CACHE_ARGS:
        arg_val = args_dict[arg]
        args_str += '#' + arg + '=' + str(arg_val)
    filename += "#" + md5(args_str) + '.' + extension
    return filename


def calculate_SIRS(aggregated_df):
    """ returns a dataframe with an additional column for SIRS score at every hour for the patient """
    # Temperature
    aggregated_df['temp_SIRS'] = 0
    aggregated_df.loc[aggregated_df['223762'] < 10, '223762'] = float("NaN")
    aggregated_df.loc[aggregated_df['223762'] > 50, '223762'] = float("NaN")
    aggregated_df.loc[aggregated_df['223761'] < 70, '223761'] = float("NaN")
    aggregated_df.loc[aggregated_df['223761'] > 120, '223761'] = float("NaN")
    aggregated_df.loc[aggregated_df['223762'] > 38, 'temp_SIRS'] = 1
    aggregated_df.loc[aggregated_df['223762'] < 36, 'temp_SIRS'] = 1
    aggregated_df.loc[aggregated_df['223761'] > 100.4, 'temp_SIRS'] = 1
    aggregated_df.loc[aggregated_df['223761'] < 96.8, 'temp_SIRS'] = 1

    # Heart rate
    aggregated_df['hr_SIRS'] = 0
    aggregated_df.loc[aggregated_df['220045'] > 300, '220045'] = float("NaN")
    aggregated_df.loc[aggregated_df['220045'] < 0, '220045'] = float("NaN")
    aggregated_df.loc[aggregated_df['220045'] > 90, 'hr_SIRS'] = 1

    # Respiratory rate
    aggregated_df['resp_SIRS'] = 0 
    aggregated_df.loc[aggregated_df['220210'] > 70, '220210'] = float("NaN")
    aggregated_df.loc[aggregated_df['220210'] < 0, '220210'] = float("NaN")
    aggregated_df.loc[aggregated_df['224690'] > 70, '224690'] = float("NaN")
    aggregated_df.loc[aggregated_df['224690'] < 0, '224690'] = float("NaN")
    aggregated_df.loc[aggregated_df['220210'] > 20, 'resp_SIRS'] = 1
    aggregated_df.loc[aggregated_df['224690'] > 20, 'resp_SIRS'] = 1

    # WBC
    aggregated_df['wbc_SIRS'] = 0 
    aggregated_df.loc[aggregated_df['51301'] > 12, 'wbc_SIRS'] = 1
    aggregated_df.loc[aggregated_df['51301'] < 4, 'wbc_SIRS'] = 1

    # Aggregation
    sirs_cols = ['temp_SIRS', 'hr_SIRS', 'resp_SIRS', 'wbc_SIRS']
    aggregated_df[sirs_cols] = aggregated_df.groupby('hadm_id')[sirs_cols].ffill().fillna(0).astype(int)

    aggregated_df['SIRS'] = aggregated_df[sirs_cols].sum(axis=1)
    aggregated_df.drop(columns=sirs_cols, inplace=True)

    return aggregated_df


def calculate_SOFA(aggregated_df):
    """ returns a dataframe with an additional column for SOFA score at every hour for the patient """
    scores = [0, 1, 2, 3, 4]
    reverse_scores = [4, 3, 2, 1, 0]

    # Respiration
    aggregated_df.loc[aggregated_df['223835'] < 1, '223835'] = aggregated_df['223835'] * 100
    aggregated_df.loc[aggregated_df['223835'] < 20, '223835'] = float("NaN")
    aggregated_df['pao2fio2ratio'] = aggregated_df['50821'] / aggregated_df['223835'] * 100
    aggregated_df['pao2fio2ratio_novent'] = aggregated_df.loc[aggregated_df['InvasiveVent']==0]['pao2fio2ratio']
    aggregated_df['pao2fio2ratio_vent'] = aggregated_df.loc[aggregated_df['InvasiveVent']==1]['pao2fio2ratio']
    
    aggregated_df['resp_SOFA'] = 0
    aggregated_df.loc[aggregated_df['pao2fio2ratio_novent'] < 400, 'resp_SOFA'] = 1
    aggregated_df.loc[aggregated_df['pao2fio2ratio_novent'] < 300, 'resp_SOFA'] = 2
    aggregated_df.loc[aggregated_df['pao2fio2ratio_vent'] < 200, 'resp_SOFA'] = 3
    aggregated_df.loc[aggregated_df['pao2fio2ratio_vent'] < 100, 'resp_SOFA'] = 4

    # Liver
    bilirubin_bins = [-1, 1.2, 2, 6, 12, float("inf")]
    aggregated_df['liver_SOFA'] = pd.cut(aggregated_df['50885'], bilirubin_bins, labels=scores).astype('float')

    # Coagulation
    coag_bins = [-1, 20, 50, 100, 150, float("inf")]
    aggregated_df['coag_SOFA'] = pd.cut(aggregated_df['51265'], coag_bins, labels=reverse_scores).astype('float')

    # Renal
    creat_bins = [-1, 1.2, 2, 3.5, 5, float("inf")]
    aggregated_df['renal_SOFA'] = pd.cut(aggregated_df['50912'], creat_bins, labels=scores).astype('float')
    urine_output_cols = ['226559', '226560', '226561', '226584', '226563', '226564', '226565', '226567',
                         '226557', '226558', '227488', '227489']
    aggregated_df.loc[aggregated_df['227488']>0, '227488'] = -aggregated_df['227488']
    aggregated_df['urine_output'] = aggregated_df[urine_output_cols].sum(axis=1)
    aggregated_df.loc[aggregated_df['urine_output'] < 500, 'renal_SOFA'] = 3
    aggregated_df.loc[aggregated_df['urine_output'] < 200, 'renal_SOFA'] = 4

    # Cardiovascular
    # features = [221662, 221653, 221289, 221906] # dopamine, dobutamine, epinephrine, norepinephrine
    aggregated_df.loc[(aggregated_df['221662_rate']>0) | (aggregated_df['221653_rate']>0), 'cardio_SOFA'] = 2
    aggregated_df.loc[(aggregated_df['221662_rate']>5) | ((aggregated_df['221289_rate'] > 0) & (aggregated_df['221289_rate']<=0.1)) | ((aggregated_df['221906_rate'] > 0) & (aggregated_df['221906_rate']<=0.1)), 'cardio_SOFA'] = 3
    aggregated_df.loc[(aggregated_df['221662_rate']>15) | (aggregated_df['221289_rate']>0.1) | (aggregated_df['221906_rate'] > 0.1), 'cardio_SOFA'] = 4

    # GCS
    # [220739, 223900, 223901] GCS-Eye, GCS-Verbal, GCS-Motor
    aggregated_df['220739'] = aggregated_df.groupby('hadm_id')['220739'].ffill().fillna(4).astype(int)
    aggregated_df['223900'] = aggregated_df.groupby('hadm_id')['223900'].ffill().fillna(5).astype(int)
    aggregated_df['223901'] = aggregated_df.groupby('hadm_id')['223901'].ffill().fillna(6).astype(int)
    aggregated_df['gcs'] = aggregated_df['220739'] + aggregated_df['223900'] + aggregated_df['223901']
    aggregated_df.loc[aggregated_df['223900'] == 0, 'gcs'] = 15

    gcs_bins = [-1, 6, 9, 12, 14, 16]
    aggregated_df['gcs_SOFA'] = pd.cut(aggregated_df['gcs'], gcs_bins, labels=reverse_scores).astype('float')
    

    # forwardfill for SOFA scores first, then replace NA's with 0. 
    sofa_cols = ['liver_SOFA', 'coag_SOFA', 'renal_SOFA', 'cardio_SOFA', 'resp_SOFA', 'gcs_SOFA']
    aggregated_df[sofa_cols] = aggregated_df.groupby('hadm_id')[sofa_cols].ffill().fillna(0).astype(int)

    aggregated_df['SOFA'] = aggregated_df[sofa_cols].sum(axis=1)
    sofa_cols = sofa_cols + ['gcs', 'urine_output']
    aggregated_df.drop(columns=sofa_cols, inplace=True)
    return aggregated_df

def calculate_SI(aggregated_df):
    """ calculates suspicion of infection as per Sepsis-3 on aggregated hourly dataframe and saves it under the column `suspicion_of_infection`.

    Note:
        aggregated_df must contain `antibiotics` and `microbio-sample` columns.
    """
    df = aggregated_df[['hadm_id', 'hour', 'antibiotics', 'microbio-sample']] # reduce data, speeds up computation
    df['antibiotics'].fillna(0, inplace=True)

    def _fix_columns(antibiotics_window_df):
        """Fixes resulting columns/index from GroupBy.rolling so that there are just hadm_id, hour, and antibiotics cols"""
        if 'hadm_id' in antibiotics_window_df.index.names and 'hadm_id' in df.columns:
            antibiotics_window_df.drop(columns='hadm_id', inplace=True)
        if 'hour' in antibiotics_window_df.index.names and 'hour' in df.columns:
            antibiotics_window_df.drop(columns='hour', inplace=True)
        antibiotics_window_df = antibiotics_window_df.reset_index()[['hadm_id', 'hour', 'antibiotics']]
        return antibiotics_window_df

    antibiotics_last_24h = df.groupby('hadm_id').rolling(on='hour', window=24, min_periods=1).antibiotics.sum()
    antibiotics_last_24h = _fix_columns(antibiotics_last_24h)
    antibiotics_last_24h = antibiotics_last_24h.rename(columns={'antibiotics': 'antibiotics_last_24h'})

    antibiotics_next_72h = df[::-1].groupby('hadm_id').rolling(on='hour', window=72, min_periods=1).antibiotics.sum()[::-1]
    antibiotics_next_72h = _fix_columns(antibiotics_next_72h)
    antibiotics_next_72h = antibiotics_next_72h.rename(columns={'antibiotics': 'antibiotics_next_72h'})

    df = df.merge(antibiotics_last_24h, on=['hadm_id', 'hour'])
    df = df.merge(antibiotics_next_72h, on=['hadm_id', 'hour'])

    microbio_sample = df['microbio-sample'] == 1

    suspicion_of_infection = microbio_sample & (df['antibiotics_last_24h'] > 0)
    suspicion_of_infection |= microbio_sample & (df['antibiotics_next_72h'] > 0)
    aggregated_df['suspicion_of_infection'] = suspicion_of_infection
    return aggregated_df

def _sepsis_sofa_diff(df, hours_before_si=48, hours_after_si=24, metric='SOFA', sepsis_col='sepsis', 
    decrease_baseline=False, sofa_diff_threshold=2, ):
    """Computes sepsis indicator labels for a single patient, by comparing SOFA score at each timestep in window around SI to 
    baseline value from first hour of window.

    Based off the following script by Moor, Michael:
    https://github.com/BorgwardtLab/mgp-tcn/blob/master/src/query/compute_sepsis_onset_from_exported_sql_table.py

    Parameters:
      - df: hourly values for patient (must contain columns 'hour', 'suspicion_of_infection', and 'SOFA' or other metric)
      - hours_before_si: defines size of window around SI
      - hours_after_si: defines size of window around SI
      - metric: column name of to check for acute increase of (SOFA or SIRS)
      - sepsis_col: which column to store sepsis flag under.
      - decrease_baseline: whether to decrease the baseline if a lower SOFA value occurs during window.
      - sofa_diff_threshold: threshold of SOFA-increase for sepsis to occur (default: 2)
    
    Note:
        Sepsis onset time is set to be the time of of SOFA-increase.
    """
    df[sepsis_col] = False # initalize to all False

    df_s = df.iloc[np.argsort(df.hour)] # sort by hour, increasing

    si_hours_df = df_s.loc[df_s.suspicion_of_infection == 1]
    si_hours = si_hours_df.hour.tolist()
    for i, si_hour in enumerate(si_hours):
        # for every SI ocurrence, calculate window around hour of SI
        si_window = df_s.loc[(si_hour-hours_before_si <= df_s.hour) & (df_s.hour <= si_hour+hours_after_si)]
        si_window["SI_hour"] = si_hour

        # check if there is an increase in SOFA during window
        sofa = np.array(si_window[metric])
        min_sofa = sofa[0]
        for i in range(len(sofa)):
            current_sofa = sofa[i]
            if decrease_baseline and current_sofa < min_sofa:
                min_sofa = current_sofa
            else:
                diff = current_sofa - min_sofa
                if diff >= sofa_diff_threshold:

                    # if there was an increase >= 2, set sepsis-time to SOFA-increase time
                    sepsis_time = si_window['hour'].iloc[i]
                    df.loc[df.hour == sepsis_time, sepsis_col] = True
                    # if there was an increase >= 2, set sepsis-time to SI-time
                    #df.loc[df.hour == si_hour, sepsis_col] = True
                    break # break to outer for-loop
    return df[sepsis_col]

def calculate_sepsis(aggregated_df, hours_before_si=48, hours_after_si=24, task='sepsis3', consider_difference=True, decrease_baseline=True):
    """ Calculates sepsis labels from hourly SOFA/SIRS and suspicion of infection.

    Note:
        Similar to other implementations, sepsis-3 is considered to happen if SOFA was >= 2 at any point in a window 
        around a suspicion of infection. Thus, it is not considered whether the SOFA increased or decreased from the start value.

    Parameters:
        - aggregated_df
        - hours_before_si: how many hours previous to the SI-time to evaluate SOFA scores for.
        - hours_after_si: how many hours after the SI-time to evaluate SOFA scores for.
        - metric:  
        - consider_difference: if true, will use slower algorithm that considers increase in metric by 2 from baseline instead of any values >= 2.

    Returns: aggregated_df with two additional columns:
        - 'sepsis': a binary label indicating times of sepsis.
        - 'sepsis_onset' a binary label only containing the first case of sepsis per-admission.
    """
    if task == 'sepsis1':
        metric = 'SIRS'
    elif task == 'sepsis3':
        metric = 'SOFA'
    else:
        raise ValueError ("Task undefined: please choose between sepsis1 and sepsis3")

    if not consider_difference:
        max_sofa_last_x_hours = aggregated_df[['hadm_id', 'hour', metric]].groupby('hadm_id').rolling(on='hour', window=hours_before_si, min_periods=1)[metric].max()
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=hours_after_si)
        max_sofa_next_y_hours = aggregated_df[['hadm_id', 'hour', metric]].groupby('hadm_id').rolling(on='hour', window=indexer, min_periods=1)[metric].max()

        df = aggregated_df[['hadm_id', 'hour', 'suspicion_of_infection']].set_index(['hadm_id', 'hour'])
        df['max_sofa_last_x_hours'] = max_sofa_last_x_hours
        df['max_sofa_next_y_hours'] = max_sofa_next_y_hours
        df.reset_index(inplace=True)

        sepsis = df['suspicion_of_infection'] & (df.max_sofa_last_x_hours >= 2)
        sepsis |= df['suspicion_of_infection'] & (df.max_sofa_next_y_hours >= 2)

        aggregated_df[task] = sepsis
    else:
        print("Computing sepsis")
        start = time.time()
        sepsis = aggregated_df[['hadm_id','hour','suspicion_of_infection',metric]].groupby("hadm_id").apply(_sepsis_sofa_diff, hours_after_si=hours_after_si, 
                                                                                                            hours_before_si=hours_before_si, metric=metric, 
                                                                                                            decrease_baseline=decrease_baseline)
        sepsis.index = sepsis.index.get_level_values(1) #drop hadm_id index to have same index as aggregated_df
        aggregated_df[task] = sepsis
        print("Took", time.time()-start, "s")

    # compute first point of sepsis3 per admission
    sepsis_onset = aggregated_df.loc[sepsis == True, ['hadm_id', 'hour', task]].sort_values(['hour']).groupby('hadm_id').first()
    sepsis_onset = sepsis_onset.rename(columns={task: task+"_onset"}).reset_index()

    aggregated_df = aggregated_df.merge(sepsis_onset, on=['hadm_id','hour'], how='left')
    aggregated_df[task+'_onset'].fillna(False, inplace=True)

    return aggregated_df
    

def create_metadata_json(static_df, timeseries_df=None):
    """Creates metadata json from dataframes
    """
    metadata_json = []
    for _, row in tqdm(static_df.iterrows(), total=len(static_df)):
        patient_json = {}
        for key, value in row.items():
            if type(value) is pd.Timestamp:
                patient_json[key] = str(value)
            else:
                patient_json[key] = value

        # add non-static features under the key 'hourly_values'
        if timeseries_df is not None:
            hourly_vals = timeseries_df.loc[timeseries_df['subject_id'] == row['subject_id']]
            assert len(hourly_vals) > 0, "Zero rows found for patient"
            patient_json['hourly_values'] = []
            cols = hourly_vals.columns[~(hourly_vals.columns == 'subject_id')]
            for _, row in hourly_vals.loc[:, cols].iterrows():
                patient_json['hourly_values'].append(dict(row.items()))

        metadata_json.append(patient_json)

    return metadata_json

MIMIC_IV_CSV_SUBFOLDER = {
    'patients': 'core',
    'admissions': 'core',
    'chartevents': 'icu',
    'icustays': 'icu',
    'inputevents': 'icu',
    'outputevents': 'icu',
    'procedureevents': 'icu',
    'labevents': 'hosp',
    'microbiologyevents': 'hosp',
    'prescriptions': 'hosp',
    'diagnoses_icd': 'hosp',
}
MIMIC_IV_CSV_DTYPES = {
    'patients': {
        'subject_id': np.uint32,
        'gender': str,
        'anchor_age': np.uint32,
        'anchor_year': str,
        'anchor_year_group': str,
        'dod': str,
    },
   'chartevents': {
        'subject_id': np.uint32,
        'hadm_id': np.uint32,
        'stay_id': np.uint32,
        'charttime': str,
        'storetime': str,
        'itemid': np.uint32,
        'value': str,
        'valuenum': float,
        'valueuom': str,
        'warning': bool
   },
    'admissions': {
        'subject_id': np.uint32,
        'hadm_id': np.uint32,
        'admittime': str,
        'dischtime': str,
        'deathtime': str,
        'admission_type': str,
        'admission_location': str,
        'discharge_location': str,
        'insurance': str,
        'language': str,
        'marital_status': str,
        'ethnicity': str,
        'edregtime': str,
        'edouttime': str,
        'hospital_expire_flag': bool,
    },
    'icustays': {
        'subject_id': np.uint32,
        'hadm_id': np.uint32,
        'stay_id': np.uint32,
        'first_careunit': str,
        'last_careunit': str,
        'intime': str,
        'outtime': str,
        'los': float
    },
    'inputevents': None,
    'labevents': None,
    'microbiologyevents': None,
    'prescriptions': None,
    'outputevents': None,
    'diagnoses_icd': None,
    'procedureevents': None,
}
MIMIC_IV_CSV_DATETIME_CONVERSION = {
    'patients': {
        'anchor_year': lambda col: pd.to_datetime(col, format="%Y"),
        'anchor_year_group': lambda col: pd.to_datetime(col.str.slice(stop=4), format="%Y"),
        'dod': pd.to_datetime
    },
    'chartevents': {
        'charttime': pd.to_datetime,
        'storetime': pd.to_datetime
    },
    'admissions': {
        'admittime': pd.to_datetime,
        'dischtime': pd.to_datetime,
        'deathtime': pd.to_datetime,
        'edregtime': pd.to_datetime,
        'edouttime': pd.to_datetime
    },
    'icustays': {
        'intime': pd.to_datetime,
        'outtime': pd.to_datetime
    },
    'inputevents': {
        'starttime': pd.to_datetime,
        'endtime': pd.to_datetime,
        'storetime': pd.to_datetime,
    },
    'labevents': {
        'charttime': pd.to_datetime,
        'storetime': pd.to_datetime
    },
    'microbiologyevents': {
        'chartdate': pd.to_datetime,
        'charttime': pd.to_datetime,
        'storedate': pd.to_datetime,
        'storetime': pd.to_datetime,
    },
    'prescriptions': {
        'starttime': pd.to_datetime,
        'stoptime': pd.to_datetime,
    },
    'outputevents': {
        'charttime': pd.to_datetime,
        'storetime': pd.to_datetime
    },
    'procedureevents': {
        'starttime': pd.to_datetime,
        'endtime': pd.to_datetime,
        'storetime': pd.to_datetime
    },
    'diagnoses_icd': {}
}

def load_data(mimic_dir: str, subset=['patients', 'chartevents', 'admissions', 'icustays'], nrows=None, chunksize=1000000, cache_dir=None) -> Sequence[pd.DataFrame]:
    """Loads MIMIC-IV Dataset with the correct types.

    Parameters:
        - mimic_dir: top-level directory of MIMIC-IV.
        - subset: names of csvs to load.
        - nrows: maximum number of rows to load from each csv.
        - chunksize: how many rows to load at once.
        - cache_dir: if set, will load and cache csvs as parquet files in this dir.

    Returns:
        - A tuple containing the specified dataframes.
    """

    mimic_dir = Path(mimic_dir) / "1.0"
    assert mimic_dir.exists(), f"{mimic_dir} does not exist!"

    result = {}

    loadingbar = tqdm(subset, leave=False)
    for csv_name in loadingbar:
        loadingbar.set_description(f"Loading {csv_name}")
        start = time.time()

        if cache_dir is not None:
            path = Path(cache_dir, csv_name + '.parquet')
            if path.is_file():
                df = pd.read_parquet(path)
                loaded_parquet = True
            else:
                loaded_parquet = False
        else:
            loaded_parquet = False

        if not loaded_parquet:
            path = Path(mimic_dir, MIMIC_IV_CSV_SUBFOLDER[csv_name], csv_name + '.csv.gz')

            df = pd.read_csv(path, dtype=MIMIC_IV_CSV_DTYPES[csv_name], chunksize=chunksize, nrows=nrows, low_memory=False)

            # if chunking, load data in chunks
            if chunksize is not None:
                data = []
                for chunk in tqdm(df, position=1, leave=False):
                    data.append(chunk)
                df = pd.concat(data, axis=0)
                del data

            # apply datetime conversion of certain columns
            for column, f in MIMIC_IV_CSV_DATETIME_CONVERSION[csv_name].items():
                df[column] = f(df[column])

        tqdm.write(f"Loaded {csv_name} ({len(df)} rows, took {time.time() - start:.2f}s)")
        if cache_dir is not None:
            parquet_path = Path(cache_dir, csv_name + '.parquet')
            if not parquet_path.is_file():
                tqdm.write(f"Writing to {str(parquet_path)}")
                df.to_parquet(parquet_path)

        result[csv_name] = df

    return result

def anchor_dates(df: pd.DataFrame, columns: Sequence[str], patients: pd.DataFrame) -> pd.DataFrame:
    """Maps dates to approximate real dates using anchor_year_group conversion.

    Note:
        Dates are mapped to the first possible real date, meaning that they may
        occur up to three years in advance of the returned date.

    Parameters:
        - df: a DataFrame containing the dates to anchor (must include a subject_id column)
        - columns: which columns of `df` to transform (these must be of type `datetime`).
        - patients: The `patients` dataframe, which contains `anchor_year` and `anchor_year_group` columns.

    Returns:
        - anchored_df - a transformed `df` where the specified columns have had their dates transformed.
    """
    # annotate df with anchor_year and anchor_year_group columns
    merge_df = pd.merge(df, patients[["subject_id", "anchor_year", "anchor_year_group"]],
                        how="left", on="subject_id")

    for column in columns:
        merge_df[column] = merge_df[column] - merge_df.anchor_year + merge_df.anchor_year_group

    return merge_df.drop(columns=["anchor_year", "anchor_year_group"])

def filter_antibiotics(prescriptions):
    """ Filters prescriptions for antibiotics that can be used for Sepsis-3 computation

    Based on the following sql script by Alistair Johnson and Tom Pollard:
    https://github.com/alistairewj/sepsis3-mimic/blob/master/query/tbls/abx-poe-list.sql
    """
    prescriptions = prescriptions.copy()
    prescriptions['route'] = prescriptions.route.str.lower()
    prescriptions['drug'] = prescriptions.drug.str.lower()

    prescriptions.dropna(subset=['drug'], inplace=True)

    # we exclude routes via the eye, ears, or topically
    mask = ~prescriptions.route.str.contains("ear").astype(bool)
    mask &= ~prescriptions.route.str.contains("eye").astype(bool)
    mask &= ~prescriptions.route.isin(('ou','os','od','au','as','ad', 'tp'))
    prescriptions = prescriptions.loc[mask]

    mask = prescriptions.drug.isin([
       'cefazolin', 'piperacillin-tazobactam', 'vancomycin',
       'sulfameth/trimethoprim ds', 'levofloxacin',
       'sulfameth/trimethoprim ss', 'amoxicillin-clavulanic acid',
       'aztreonam', 'azithromycin ', 'metronidazole (flagyl)',
       'piperacillin-tazobactam na', 'ampicillin-sulbactam',
       'doxycycline hyclate', 'nitrofurantoin monohyd (macrobid)',
       'cefepime', 'ceftazidime', 'amoxicillin', 'clarithromycin',
       'azithromycin', 'ciprofloxacin hcl', 'tobramycin sulfate',
       'clindamycin', 'cephalexin', 'metronidazole', 'ampicillin sodium',
       'ciprofloxacin iv', 'vancomycin intraventricular',
       'vancomycin oral liquid', 'cefpodoxime proxetil', 'gentamicin',
       'nitrofurantoin (macrodantin)', 'vancomycin enema',
       'amoxicillin oral susp.', 'clindamycin solution', 'minocycline',
       'ceftolozane-tazobactam', 'erythromycin',
       'amoxicillin-clavulanate susp.',
       'sulfameth/trimethoprim suspension', 'dicloxacillin',
       'vancomycin antibiotic lock', 'sulfameth/trimethoprim', 'amikacin',
       'ampicillin', 'gentamicin sulfate', 'trimethoprim',
       'tetracycline hcl', 'moxifloxacin',
       'sulfamethoxazole-trimethoprim', 'sulfadiazine',
       'ceftazidime antibiotic lock', 'penicillin v potassium',
       'penicillin g benzathine', 'penicillin g potassium', 'avelox',
       'rifampin', 'tetracycline', 'ery-tab',
       'erythromycin ethylsuccinate suspension', 'ciprofloxacin',
       'doxycycline', 'bactrim', 'vancomycin ', 'amikacin inhalation',
       'penicillin g k graded challenge', 'cefadroxil',
       'tobramycin inhalation soln', 'vancocin',
       'cefepime graded challenge', 'ceftolozane-tazobactam *nf*',
       'ceftazidime graded challenge',
       'piperacillin-tazo graded challenge', 'augmentin suspension',
       'nitrofurantoin macrocrystal',
       'ampicillin-sulbact graded challenge', 'clindamycin suspension',
       'ceftazidime-avibactam *nf*', 'augmentin',
       'ampicillin graded challenge', 'doxycycline hyclate  20mg',
       'clindamycin phosphate', 'cefdinir', 'gentamicin (bulk)',
       'streptomycin sulfate', 'vancomycin intrathecal',
       'ceftazidime-avibactam (avycaz)', 'nitrofurantoin ', 'cefpodoxime',
       'oxacillin', 'cipro', '*nf* moxifloxacin', 'flagyl',
       'nitrofurantoin', 'levofloxacin graded challenge',
       'tobramycin with nebulizer', 'keflex', 'chloramphenicol na succ',
       'tobramycin in 0.225 % nacl', 'ciprofloxacin ',
       'doxycycline monohydrate', 'vancomycin 125mg cap',
       'vancomycin ora', 'gentamicin antibiotic lock', 'cefotaxime',
       'ciproflox', 'amoxicillin-clavulanate susp',
       'amoxicillin-pot clavulanate', 'gentamicin intraventricular',
       'gentamicin 2.5 mg/ml in sodium citrate 4%',
       'sulfameth/trimethoprim ', 'trimethoprim-sulfamethoxazole',
       'cefuroxime axetil', 'vancomycin 250 mg', 'tobramycin',
       'levofloxacin 100mg/4ml solution', 'macrodantin',
       'rifampin 150mg capsules', 'cefoxitin', '*nf* cefoxitin sodium',
       'ampicillin-sulbactam sodium', 'doxycycline ', 'bactrim ',
       'bactrim ds', 'neo*iv*gentamicin', 'neo*iv*oxacillin',
       'neo*iv*vancomycin', 'neo*iv*penicillin g potassium',
       'neo*iv*cefotaxime', 'trimethoprim oral soln',
       'cephalexin suspension', 'penicillin ', 'neo*iv*cefazolin',
       'levofloxacin ', 'neo*iv*ceftazidime', 'neo*po*azithromycin',
       'erythromycin ethylsuccinate', 'zithromax z-pak',
       'vancomycin for inhalation', 'vancomycin for nasal inhalation',
       'penicillin v potassium suspension', 'vancocin (vancomycin)',
       'minocycline 100mg tablets', 'clindamycin  cap',
       'cefpodoxime 200mg tab', 'clindamycin hcl caps', 'clindamycin hcl',
       'nitrofurantoin monohyd/m-cryst', 'nitrofurantoin macrocrystals',
       'nitrofurantoin macrocystals', 'vancomycin capsule',
       '*nf* cefuroxime', 'vancomycin oral capsule', 'vancomycin caps',
       'erythromycin ', 'azithromycin po susp', 'cayston',
       'vancomycin 250mg', 'cefotaxime ', 'vancomycin-heparin lock',
       'amoxicillin-clavulanate po susp 400 mg-57 mg/5 ml',
       'penicillin v potassium solution', 'inv-tivantinib',
       'cefazolin 2 g'])

    prescriptions = prescriptions.loc[mask]
    return prescriptions

def filter_diagnoses(diagnoses_icd: pd.DataFrame, item_map: pd.DataFrame):
    item_map = item_map.loc[item_map.origin == 'diagnoses_icd']
    result = []
    for icd_version in set(item_map.icd_version):
        remaining = diagnoses_icd.loc[diagnoses_icd.icd_version == icd_version]
        icd_version_codes = set(item_map.loc[item_map.icd_version == icd_version].itemid)
        for code in icd_version_codes:
            mask = diagnoses_icd.icd_code.str.startswith(code, na=False)
            filter_result = remaining.loc[mask]
            filter_result['icd_code'] = code
            result.append(filter_result)
            remaining = remaining.loc[~mask]

    return pd.concat(result)

def filter_table_patients(df, patients):
    subject_ids = set(patients.subject_id.unique())
    df.set_index('subject_id', inplace=True) # set index to speed up filtering
    df = df.loc[df.index.isin(subject_ids)]
    df.reset_index(inplace=True)
    return df


def filter_eligible_patients(admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame, 
                             min_patient_age:int=15, min_hours:int=24, gap_hours:int=6, min_icu_stay:int=12, max_icu_stay:int=10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Filters the admissions dataframe to only include eligible criteria:
        - Patient is above the age of min_patient_age
        - Hospital stay (hadm_id) has greater than min_hours + gap_hours hours of data
        - First ICU stay for the subject
        - ICU stay between min_icu_stay hours and max_icu_stay days
    
    Returns
        - admissions - a filtered version of admissions
        - patients - a filtered version of patients
        - first_icustays - a filtered version of icustayss
    """
    print("Filtering patient age...")
    patients = patients.loc[patients["anchor_age"]>min_patient_age]

    subject_ids = set(patients.subject_id.unique())
    
    print("Filtering hospital stay time...")
    admissions = admissions.loc[admissions['dischtime']-admissions['admittime'] > pd.Timedelta(hours=min_hours+gap_hours)]

    print("Filtering ICU stay time...")
    # update icustays to only include the patients that have been filtered in the previous steps
    subject_ids = subject_ids.intersection(set(admissions['subject_id'].unique()))
    icustays = icustays.loc[icustays["subject_id"].isin(subject_ids)]

    first_icustays = icustays.loc[icustays.groupby("subject_id")["intime"].idxmin()]
    first_icustays = first_icustays.loc[first_icustays['outtime']-first_icustays['intime'] > pd.Timedelta(hours=min_icu_stay)]
    first_icustays = first_icustays.loc[first_icustays['outtime']-first_icustays['intime'] < pd.Timedelta(days=max_icu_stay)]

    hadm_ids = set(first_icustays.hadm_id).intersection(set(admissions.hadm_id))
    first_icustays = first_icustays.loc[first_icustays.hadm_id.isin(hadm_ids)]

    subject_ids = subject_ids.intersection(set(first_icustays.subject_id.unique()))
    admissions = admissions.loc[admissions.subject_id.isin(subject_ids)]
    patients = patients.loc[patients.subject_id.isin(subject_ids)]

    return admissions, patients, first_icustays


def filter_events_to_timerange(chartevents, hours_from_start): #!TODO: make generic column to use for inputevents
    first_charttime = chartevents.groupby("subject_id")["charttime"].min()
    time_since_first = chartevents['charttime'] - first_charttime.loc[chartevents['subject_id']].reset_index(drop=True)
    chartevents = chartevents.loc[time_since_first < pd.Timedelta(hours=hours_from_start)]
    return chartevents

def filter_variables(chartevents, items):
    """ Takes in a table like chartevents, and a list `items` that contains all the item_id's to keep. 

    Returns the filtered chartevents with the right variables
    """
    return chartevents.loc[chartevents['itemid'].isin(items)]

def filter_events_to_stay(chartevents, static_df):
     # filter out relevant hospital stay data
    mask = chartevents.hadm_id.isin(static_df.hadm_id)
    events = chartevents[mask]

    to_drop = [col for col in ['subject_id', 'stay_id'] if col in events]
    events.drop(columns = to_drop, inplace=True)
    events = pd.merge(events, static_df, on='hadm_id', how='inner')
    events = events.loc[events['charttime'] > events['intime']]
    events = events.loc[events['charttime'] < events['outtime']]
    filtered_events = events.drop(columns=static_df.columns.drop(['hadm_id', 'subject_id']))

    return filtered_events

def add_vents(aggregated_df, vents_df):
    """ takes in a df aggregated by hour, and a df of vent times by patient (already filtered for stay)
    returns a new column ['InvasiveVent'] for each hour a patient is on invasive vent""" 

    vents_df = vents_df.loc[vents_df['ventilation_status'] == 'InvasiveVent']

    vents_df = vents_df[['hadm_id', 'starttime', 'endtime']].groupby('hadm_id')
    n_vents_max = vents_df.starttime.count().max()

    merged_df = aggregated_df
    merged_df['InvasiveVent'] = 0
    for i in range(n_vents_max):
        vent_df = vents_df.nth(i)
        vent_df.reset_index(inplace=True)

        merged_df = pd.merge(merged_df, vent_df, on='hadm_id', how='left')
        merged_df.loc[(merged_df['charttime']+pd.Timedelta(hours=1)>= merged_df['starttime']) \
            & (merged_df['charttime'] <= merged_df['endtime']), 'InvasiveVent'] = 1
        
        merged_df.drop(columns=['starttime', 'endtime'], inplace=True)

    return merged_df

def hourly_aggregation(events, static_df, inputevents=None, antibiotics=None, microbio=None, outputevents=None, procedureevents=None, prescriptions=None): 
    # remove superflous columns
    events = events[['hadm_id', 'charttime', 'itemid', 'valuenum']]

    # if prescriptions exist, merge with events
    if antibiotics is not None:
        antibiotics['itemid'] = 'antibiotics'
        antibiotics['valuenum'] = 1
        antibiotics = antibiotics[events.columns]
        events = pd.concat([antibiotics, events])
    
    if microbio is not None: 
        microbio['itemid'] = 'microbio-sample'
        microbio['valuenum'] = 1
        microbio = microbio[events.columns]
        events = pd.concat([microbio, events])


    # aggregate items by hour of icu stay
    pivot_events = events.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values = 'valuenum')
    pivoted_df = pivot_events

    # pivot inputevents on starttime and hadm_id, itemid, values = amount & rate
    # combine inputevents with pivoted other items
    if inputevents is not None: 
        pivot_inputvals = inputevents.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values = 'amount')
        pivot_inputrate = inputevents.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values = 'rate')
        pivot_inputrate = pivot_inputrate.add_suffix('_rate')

        pivoted_df = pivoted_df.join(pivot_inputvals, how='outer')
        pivoted_df = pivoted_df.join(pivot_inputrate, how='outer')

    if outputevents is not None:
        pivot_outputvals = outputevents.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values = 'value')
        pivoted_df = pivoted_df.join(pivot_outputvals, how='outer')

    if procedureevents is not None:
        pivot_procedures = procedureevents.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values = 'value')
        pivoted_df = pivoted_df.join(pivot_procedures, how='outer')
    
    if prescriptions is not None: 
        prescriptions['valuenum'] = 1
        pivot_prescriptions = prescriptions.pivot_table(index=['hadm_id', 'charttime'], columns='itemid', values='valuenum')
        pivoted_df = pivoted_df.join(pivot_prescriptions, how='outer')

    pivoted_df.reset_index(inplace=True)
    pivoted_df = pivoted_df.merge(static_df[['intime', 'hadm_id']], on='hadm_id')

    def add_intime_to_group(group):
        group = group.append(pd.Series(), ignore_index=True)
        group.loc[group.index[-1], 'charttime'] = group.iloc[0]['intime']
        return group

    pivoted_df = pivoted_df.groupby('hadm_id').apply(lambda x: add_intime_to_group(x))
    pivoted_df.drop(columns=['hadm_id'], inplace=True)
    pivoted_df.reset_index(inplace=True)
    if 'level_1' in pivoted_df:
        pivoted_df.drop(columns=['level_1'], inplace=True)

    # resample combined table, if inputevents then take sum of those, take mean of others
    

    if inputevents is not None: 
        input_cols = list(pivot_inputvals.columns) + list(pivot_outputvals.columns) + list(pivot_procedures.columns) + ['hadm_id', 'charttime']
        event_cols = list(pivot_inputrate.columns)+ list(pivot_events.columns) + list(pivot_prescriptions.columns) + ['hadm_id', 'charttime']

        resample_input = pivoted_df[input_cols].groupby('hadm_id').resample('H', on='charttime', origin='start').sum()
        resample_events = pivoted_df[event_cols].groupby('hadm_id').resample('H', on='charttime', origin='start').mean()

        for df in [resample_input, resample_events]:
            if 'hadm_id' in df:
                df.drop(columns=['hadm_id'], inplace=True)

        resample = resample_input.join(resample_events, how='outer')

    else:
        resample = pivoted_df.groupby('hadm_id').resample('H', on='charttime', origin='start').mean() 
        if 'hadm_id' in resample:
            resample.drop(columns=['hadm_id'], inplace=True)

    resample.reset_index(inplace=True)
    resample['hour'] = resample.groupby('hadm_id')['charttime'].rank(method='first', ascending=True)

    columns = list(resample.columns[2:-1])
    columns = ['hadm_id', 'hour', 'charttime'] + columns
    final_aggregation = resample[columns]
    final_aggregation.columns = final_aggregation.columns.map(str)

    return final_aggregation

def flatten_timeseries(aggregated_df):
    """
    """
    if 'charttime' in aggregated_df.columns:
        aggregated_df = aggregated_df.drop(columns='charttime')
    return pd.pivot(aggregated_df, index='hadm_id', columns='hour')

def compute_timeseries_moments(aggregated_df, moments=['mean']):
    """Summarizes all hours of timeseries data for a given feature with the specified moments.
    
    Parameters:
        - aggregated_df: dataframe of shape n_patients*hours x k_features
        - moments: list containing one or multiple of 'mean', 'var', 'skew', and/or 'kurt'
    Returns:
        - dataframe of shape n_patients x k_features
    """
    if 'charttime' in aggregated_df.columns:
        aggregated_df = aggregated_df.drop(columns='charttime')
    if 'hour' in aggregated_df.columns:
        aggregated_df = aggregated_df.drop(columns='hour')

    grouped_df = aggregated_df.groupby('hadm_id')
    
    results = []
    for moment_str in moments:
        if moment_str == 'mean':
            result = grouped_df.mean()
        elif moment_str == 'var':
            result = grouped_df.var()
        elif moment_str == 'skew':
            result = grouped_df.skew()
        else:
            assert moment_str == 'kurt', "Tried to calculate invalid moment '{}'!".format(moment_str)
            result = grouped_df.kurt()

        result.rename(lambda col: col+'_'+moment_str, axis='columns', inplace=True)
        results.append(result)
    
    return pd.concat(results, axis=1)


def reindex_timeseries(hourly_df):
    """Creates new rows for patients with less than max hours of data, so every patient matches
       length. New values are NaNs. 
    """
    if 'icustay_id' in hourly_df:
        hourly_df.drop(columns='icustay_id', inplace=True)
    if 'subject_id' in hourly_df:
        hourly_df.drop(columns='subject_id', inplace=True)

    new_index = pd.MultiIndex.from_product([hourly_df['hadm_id'].unique(), hourly_df['hour'].unique()])
    hourly_df = hourly_df.set_index(['hadm_id', 'hour']).reindex(new_index)
    hourly_df = hourly_df.reset_index().rename(columns={'level_0': 'hadm_id', 'level_1': 'hour'})
    hourly_df = hourly_df.sort_values(['hadm_id', 'hour'])
    return hourly_df


def impute_timeseries(hourly_df, method='zeroes', feature_cols=None):
    """ Fills in NaN values of dataframe. 
        Implements "simple imputation" (see: Che et al., ‘Recurrent Neural Networks for Multivariate Time Series with Missing Values’.)
    
    Parameters:
        - hourly_df: DataFrame containing hourly measurements for patients.
        - method: one of 'zeroes', 'forward', or 'simple'.
    """
    if method == 'zeroes':
        hourly_df.fillna(value=0, inplace=True)

    elif method == 'forward':
        assert feature_cols is not None, "Needs feature_cols!"
        hourly_df[feature_cols] = hourly_df.groupby('hadm_id')[feature_cols].ffill()
        hourly_df[feature_cols] = hourly_df[feature_cols].fillna(0)

    else:
        assert method == 'simple', "Invalid imputation method specified!"
        assert feature_cols is not None, "Needs feature_cols!"
        
        df = hourly_df.copy()

        for c in feature_cols:
            df[c+'_exist'] = df[c].notnull()*1
            df[c+'_time_since'] = None
            
            is_absent = pd.concat([df['hadm_id'], 1 - df[c+'_exist']], axis=1)
            hours_absent = is_absent.groupby('hadm_id')[c+'_exist'].cumsum()

            absent_times = pd.concat([df['hadm_id'], hours_absent.copy()], axis=1)
            absent_times[is_absent==1] = None

            f_absent_times = absent_times.groupby('hadm_id').ffill()

            time_since_measured = hours_absent - f_absent_times[c+'_exist']
            df[c+'_time_since'] = time_since_measured.fillna(100)

        df[feature_cols] = df.groupby('hadm_id')[feature_cols].ffill()
        df[feature_cols] = df.groupby('hadm_id')[feature_cols].fillna(0)

        hourly_df = df

    return hourly_df

def standardize_units(chartevents, itemid_to_label_map):
    """Standardize units in the same way as MIMIC-Extract.
    """
    
    # convert fahrenheit to celsius
    chartevents.loc[chartevents['valueuom'] == '°F', 'valuenum'] = (chartevents['valuenum'] - 32) * 5./9

    # convert pounds to kg
    chartevents.loc[chartevents['valueuom'] == 'lbs', 'valuenum'] = chartevents['valuenum']*0.45359237
    
    # inches not present, so no need to convert them

    # if fraction inspired oxygen > 1 divide by 100
    # fio_mask = labeled_df['label'] == 'inspired o2 fraction'
    # fio_mask &= labeled_df['valuenum'] > 1
    # labeled_df.loc[fio_mask, 'valuenum'] /= 100

    # if oxygen saturation <=1 multiply by 100 -- id's 220277 and 220227 
    chartevents.loc[(chartevents['itemid']==220277) & (chartevents['valuenum']< 1), 'valuenum']*=100
    chartevents.loc[(chartevents['itemid']==220227) & (chartevents['valuenum']< 1), 'valuenum']*=100


    return chartevents

def extract_data_prior_to_event(df, events, key, df_hour_column='hour', events_hour_column='hour', gap_hours=1, data_hours=7, case_control=False, dascena_control=False):
    """ filters df to only contain data from a certain amount of hours prior to some events.
        If there are no events, then a random point in time is chosen to extract data from.
    
    Parameters:
        - df: dataframe to filter
        - events: dataframe containing keys (e.g. patient ids) and hours of when the events occurred.
        - key: column that joins 'events' and 'df' together (e.g. patient id).
        - df_hour_column: 
        - events_hour_column: 
        - gap_hours: 
        - data_hours: 
    """
    has_events = df[key].isin(events[key].unique())
    control_keys = df.loc[~has_events, key].unique()

    # filter out patients with onset before gap time
    if dascena_control:
        print("Number of patients filtered out from too early sepsis onset:", 
            (events[events_hour_column] <= gap_hours).sum())
        events = events.loc[events[events_hour_column] > gap_hours]
        
    else:
        print("Number of patients filtered out from too early sepsis onset:", 
            (events[events_hour_column] <= gap_hours + data_hours).sum())
        events = events.loc[events[events_hour_column] > gap_hours + data_hours]
    event_keys = events[key].unique()

    np.random.seed(0)

    # Code from Borgward match_controls.py
    if case_control:
        result = pd.DataFrame()

        ratio = len(control_keys)/float(len(event_keys))
        rf = int(np.floor(ratio))
        print("Control ratio used:", rf)
        controls = df.loc[df[key].isin(control_keys)][[key]].drop_duplicates()

        controls_s = controls.iloc[np.random.permutation(len(controls))] #Shuffle controls dataframe rows, for random control selection

        for i, case_id in tqdm(enumerate(event_keys), desc='Assigning control samples', total=len(event_keys)):
            matched_controls = controls_s.iloc[(i*rf):(rf*(i+1))][key].unique() #select the next batch of controls to match to current case
            onset_hour = float(events.loc[events[key]==case_id][events_hour_column]) # get float of current case onset hour
            matched_data = controls.loc[controls[key].isin(matched_controls)][[key]]
            matched_data[events_hour_column] = onset_hour #use sepsis_onset_hour of current case as control_onset_hour
            
            result = result.append(matched_data, ignore_index=True)

        events = pd.concat((events, result))

    # choose a random point in time as the event-time for cases without events
    else:
        max_hours = df.groupby(key)[df_hour_column].max()
        max_hours = max_hours.reset_index()
        new_events = max_hours.loc[max_hours[key].isin(control_keys)].copy()
        if dascena_control:
            new_events[events_hour_column] = np.random.randint(gap_hours+1, new_events[df_hour_column]+1)
        else:
            new_events[events_hour_column] = np.random.randint(gap_hours+data_hours+1, new_events[df_hour_column]+1)
        events = pd.concat((events, new_events))

    df = df.loc[df[key].isin(events[key])]
    df = df.merge(events[[key, events_hour_column]], on=key, how='left')

    # filter data to timerange, and also filter out sepsis cases within gap_hours
    hours_to_event = df[events_hour_column] - df[df_hour_column]
    hours_mask = hours_to_event >= gap_hours
    hours_mask &= hours_to_event < gap_hours + data_hours

    df = df.loc[hours_mask]
    df['datapoints'] = df.groupby('hadm_id')['hour'].transform("count")
    df['hour'] = df.groupby('hadm_id')['hour'].rank('first') + data_hours - df['datapoints'] # right-pad the hours
    return df
