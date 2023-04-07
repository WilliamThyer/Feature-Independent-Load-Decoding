from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import time
import itertools
from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import scipy.stats as sista
from statsmodels.stats.multitest import multipletests


class Experiment:
    def __init__(self, experiment_name, data_dir, info_from_file=True, test=False, info_variable_names=['unique_id', 'chan_labels', 'chan_x', 'chan_y', 'chan_z', 'sampling_rate', 'times']):

        self.experiment_name = experiment_name
        self.data_dir = Path(data_dir)

        self.xdata_files = list(self.data_dir.glob('*xdata*.mat'))
        self.ydata_files = list(self.data_dir.glob('*ydata*.mat'))
        if test:
            self.xdata_files = self.xdata_files[0:3]
            self.ydata_files = self.ydata_files[0:3]
        self.nsub = len(self.xdata_files)

        self.behavior_files = None
        self.artifact_idx_files = None
        self.info_files = None

        if info_from_file:
            self.info = self.load_info(0, info_variable_names)
            # self.info.pop('unique_id')

    def load_eeg(self, isub):
        subj_mat = sio.loadmat(
            self.xdata_files[isub], variable_names=['xdata'])
        xdata = np.moveaxis(subj_mat['xdata'], [0, 1, 2], [1, 2, 0])
        # xdata = subj_mat['xdata']

        subj_mat = sio.loadmat(
            self.ydata_files[isub], variable_names=['ydata'])
        ydata = np.squeeze(subj_mat['ydata'])

        return xdata, ydata

    def load_ydata(self, isub):
        subj_mat = sio.loadmat(
            self.ydata_files[isub], variable_names=['ydata'])
        ydata = np.squeeze(subj_mat['ydata'])

        return ydata

    def load_behavior(self, isub, remove_artifact_trials=True):
        """
        returns behavior from csv as dictionary

        remove_artifact_trials will remove all behavior trials that were excluded from EEG data due to artifacts
        """
        if not self.behavior_files:
            self.behavior_files = list(self.data_dir.glob('*.csv'))
        behavior = pd.read_csv(self.behavior_files[isub]).to_dict('list')

        if remove_artifact_trials:
            artifact_idx = self.load_artifact_idx(isub)
            for k in behavior.keys():
                behavior[k] = np.array(behavior[k])[artifact_idx]
        else:
            for k in behavior.keys():
                behavior[k] = np.array(behavior[k])

        return behavior

    def load_artifact_idx(self, isub):
        """
        returns artifact index from EEG artifact rejection

        useful for removing behavior trials not included in EEG data
        """
        if not self.artifact_idx_files:
            self.artifact_idx_files = list(
                self.data_dir.glob('*artifact_idx*.mat'))
        artifact_idx = np.squeeze(sio.loadmat(
            self.artifact_idx_files[isub])['artifact_idx'] == 1)

        return artifact_idx

    def load_info(self, isub, variable_names=['unique_id', 'chan_labels', 'chan_x', 'chan_y', 'chan_z', 'sampling_rate', 'times']):
        """ 
        loads info file that contains data about EEG file and subject
        """
        if not self.info_files:
            self.info_files = list(self.data_dir.glob('*info*.mat'))
        info_file = sio.loadmat(
            self.info_files[isub], variable_names=variable_names)
        info = {k: np.squeeze(info_file[k]) for k in variable_names}

        return info


class Experiment_Syncer:
    def __init__(
        self,
        experiments,
        wrangler,
        train_group,
        get_matched_data=True
    ):
        self.experiments = experiments
        self.wrangler = wrangler
        self.train_group = train_group
        self.experiment_names = []
        for i in range(len(experiments)):
            self.experiment_names.append(experiments[i].experiment_name)

        self._load_unique_ids()
        if get_matched_data:
            self._find_matched_ids()
        else:
            self._find_all_ids()

    def _load_unique_ids(self):
        '''
        Loads all IDs in all experiments
        '''
        self.all_ids = []
        for exp in self.experiments:
            exp.unique_ids = []
            for isub in range(exp.nsub):
                exp.unique_ids.append(int(exp.load_info(isub)['unique_id']))
            self.all_ids.extend(exp.unique_ids)
        self.all_ids = np.unique(self.all_ids)

        self.matched_ids = []
        for i in self.all_ids:
            check = 0
            for exp in self.experiments:
                if i in exp.unique_ids:
                    check += 1
            if check == len(self.experiments):
                self.matched_ids.append(i)

    def _find_matched_ids(self):
        '''
        Finds only IDs that are in all experiments
        '''
        self.id_dict = dict.fromkeys(self.matched_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.matched_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(
                        m)
                except ValueError:
                    pass
        self.nsub = len(self.matched_ids)

    def _find_all_ids(self):
        '''
        Finds IDs in all experiments. Used for loading all data across experiments.
        '''
        self.id_dict = dict.fromkeys(self.all_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.all_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(
                        m)
                except ValueError:
                    pass
        self.nsub = len(self.all_ids)

    def load_eeg(self, sub):
        xdata = dict.fromkeys(self.experiment_names)
        ydata = dict.fromkeys(self.experiment_names)
        for exp in self.experiments:
            if self.id_dict[sub][exp.experiment_name] is not None:
                xdata[exp.experiment_name], ydata[exp.experiment_name] = exp.load_eeg(
                    self.id_dict[sub][exp.experiment_name])
            else:
                xdata.pop(exp.experiment_name)
                ydata.pop(exp.experiment_name)
        return xdata, ydata

    def load_behavior(self, sub):
        beh = dict.fromkeys(self.experiment_names)
        for exp in self.experiments:
            if self.id_dict[sub][exp.experiment_name] is not None:
                beh[exp.experiment_name] = exp.load_behavior(
                    self.id_dict[sub][exp.experiment_name])
            else:
                beh.pop(exp.experiment_name)
        return beh

    def select_labels(self, xdata, ydata):
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.select_labels(
                xdata[exp_name], ydata[exp_name])
        return xdata, ydata

    def group_labels(self, xdata, ydata):
        '''
        groups classes based on group_dict, removes not-included classes
        '''
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.group_labels(
                xdata[exp_name], ydata[exp_name])
        return xdata, ydata

    def balance_labels(self, xdata, ydata):
        '''
        balances number of class instances
        '''
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.balance_labels(
                xdata[exp_name], ydata[exp_name])
        return xdata, ydata

    def average_trials(self, xdata, ydata):
        '''
        bins trials based on trial_average
        '''
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.average_trials(
                xdata[exp_name], ydata[exp_name])
        return xdata, ydata

    def setup_data(self, xdata, ydata, labels=False, group_dict=False):
        '''
        does basic data manipulation using other functions.
        '''
        if labels:
            xdata, ydata = self.select_labels(xdata, ydata)
        if group_dict:
            xdata, ydata = self.group_labels(xdata, ydata)
        xdata, ydata = self.balance_labels(xdata, ydata)
        xdata, ydata = self.average_trials(xdata, ydata)
        return xdata, ydata

    def pairwise(self, xdata_all, ydata_all):
        '''
        When using group_dict_list (e.g. 1vs2 then 2vs4), yields data with only those classes.
        '''

        for self.wrangler.iss, ss in enumerate(self.wrangler.group_dict_list):
            xdata, ydata = deepcopy(xdata_all), deepcopy(ydata_all)

            self.wrangler.group_dict = ss

            for exp_name in xdata.keys():
                xdata[exp_name], ydata[exp_name] = self.wrangler.group_labels(
                    xdata[exp_name], ydata[exp_name])
            yield xdata, ydata

    def group_data(self, xdata, ydata):
        '''
        groups data into train and test groups based on self.train_group
        '''
        xdata_train, xdata_test = None, None

        for exp_name in xdata.keys():

            if np.isin(exp_name, self.train_group):
                if xdata_train is not None:
                    xdata_train = np.append(xdata_train, xdata[exp_name], 0)
                    ydata_train = np.append(ydata_train, ydata[exp_name], 0)
                elif xdata_train is None:
                    xdata_train = xdata[exp_name]
                    ydata_train = ydata[exp_name]
            else:
                if xdata_test is not None:
                    xdata_test = np.append(xdata_test, xdata[exp_name], 0)
                    ydata_test = np.append(ydata_test, ydata[exp_name], 0)
                elif xdata_test == None:
                    xdata_test = xdata[exp_name]
                    ydata_test = ydata[exp_name]
        if xdata_test is None:  # if both groups are in train_group, function combines and returns as one
            return xdata_train, ydata_train
        else:
            return xdata_train, xdata_test, ydata_train, ydata_test


class Wrangler:
    def __init__(self,
                 samples,
                 time_window, time_step,
                 trial_average,
                 n_splits,
                 group_dict=None,
                 group_dict_list=None,
                 labels=None,
                 electrodes=None,
                 electrode_subset_list=None):

        self.samples = samples
        self.sample_step = samples[1]-samples[0]
        self.time_window = time_window
        self.time_step = time_step
        self.trial_average = trial_average
        self.n_splits = n_splits
        self.group_dict = group_dict
        self.group_dict_list = group_dict_list
        self.labels = labels
        self.electrodes = electrodes
        self.electrode_subset_list = electrode_subset_list

        if self.group_dict_list:
            self.labels = []
            self.num_labels = []
            for group_dict in group_dict_list:
                labels = [j for i in list(group_dict.values()) for j in i]
                self.labels.append(labels)
                self.num_labels.append(len(labels))
        else:
            if self.group_dict:
                self.labels = [j for i in list(group_dict.values()) for j in i]
            if self.labels:
                self.num_labels = len(self.labels)
            else:
                self.num_labels = None

        self.cross_val = StratifiedShuffleSplit(n_splits=self.n_splits)

        self.t = samples[0:samples.shape[0] -
                         int(time_window/self.sample_step)+1:int(time_step/self.sample_step)]

    def select_labels(self, xdata, ydata, return_idx=True):
        """
        includes labels only wanted for decoding

        returns xdata and ydata with unwanted labels removed

        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """

        label_idx = np.isin(ydata, self.labels)
        xdata = xdata[label_idx, :, :]
        ydata = ydata[label_idx]

        if return_idx:
            return xdata, ydata, label_idx
        else:
            return xdata, ydata

    def group_labels(self, xdata, ydata, empty_val=9999):
        """
        groups classes based on group dict. Also excludes classes not included in group_dict.
        If one of your class labels is 9999, change empty_val to something your class label isn't.

        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """

        xdata_new = np.ones(xdata.shape)*empty_val
        ydata_new = np.ones(ydata.shape)*empty_val

        for k in self.group_dict.keys():
            trial_idx = np.arange(ydata.shape[0])[
                np.isin(ydata, self.group_dict[k])]
            xdata_new[trial_idx] = xdata[trial_idx]
            ydata_new[trial_idx] = k

        trial_idx = ydata_new == empty_val
        return xdata_new[~trial_idx], ydata_new[~trial_idx]

    def pairwise(self, xdata_all, ydata_all):
        '''
        When using group_dict_list (e.g. 1vs2 then 2vs4), yields data with only those classes.
        '''
        for self.iss, ss in enumerate(self.group_dict_list):
            xdata, ydata = deepcopy(xdata_all), deepcopy(ydata_all)
            self.group_dict = ss
            yield self.group_labels(xdata, ydata)

    def balance_labels(self, xdata, ydata, downsamp=None):
        '''
        balances number of class instances
        '''
        unique_labels, counts_labels = np.unique(ydata, return_counts=True)
        if downsamp is None:
            downsamp = min(counts_labels)
        label_idx = []
        for label in unique_labels:
            label_idx = np.append(label_idx, np.random.choice(
                np.arange(len(ydata))[ydata == label], downsamp, replace=False))

        xdata = xdata[label_idx.astype(int), :, :]
        ydata = ydata[label_idx.astype(int)]

        return xdata, ydata

    def average_trials(self, xdata, ydata):
        '''
        bins trials based on trial_average
        '''
        if self.trial_average:
            unique_labels, counts_labels = np.unique(ydata, return_counts=True)
            min_count = np.floor(min(counts_labels) /
                                 self.trial_average)*self.trial_average
            nbin = int(min_count/self.trial_average)
            trial_groups = np.tile(np.arange(nbin), self.trial_average)

            xdata_new = np.zeros(
                (nbin*len(unique_labels), xdata.shape[1], xdata.shape[2]))
            count = 0
            for ilabel in unique_labels:
                for igroup in np.unique(trial_groups):
                    xdata_new[count] = np.mean(xdata[ydata == ilabel][:int(
                        min_count)][trial_groups == igroup], axis=0)
                    count += 1
            ydata_new = np.repeat(unique_labels, nbin)
            return xdata_new, ydata_new
        else:
            return xdata, ydata

    def setup_data(self, xdata, ydata, balance=True):
        '''
        does basic data manipulation using other functions.
        '''
        if self.group_dict:
            xdata, ydata = self.group_labels(xdata, ydata)
        elif self.labels:
            xdata, ydata = self.select_labels(xdata, ydata)
        if balance is True:
            xdata, ydata = self.balance_labels(xdata, ydata)
        xdata, ydata = self.average_trials(xdata, ydata)
        return xdata, ydata

    def select_electrodes(self, xdata, electrode_subset=None):
        '''
        removes electrodes not included in electrode_subset.
        Not used in analyses for paper.
        '''
        if electrode_subset is not None:
            # Create index for electrodes to include in plot
            electrode_labels = [el for n, el in enumerate(
                self.electrodes) if el.startswith(electrode_subset)]
            electrode_idx = np.in1d(self.electrodes, electrode_labels)

        return xdata[:, electrode_idx]

    def roll_over_electrodes(self, xdata_all, ydata_all, electrode_subset_list=None, electrode_idx_list=None):
        '''
        yields data with electrodes not in electrode subset, iterating over electrode_subset_list.
        Not used in analyses for paper.
        '''
        for self.ielec, electrode_subset in enumerate(self.electrode_subset_list):
            yield self.select_electrodes(xdata_all, electrode_subset), ydata_all

    def train_test_split(self, xdata, ydata, return_idx=False):
        """
        returns xtrain and xtest data and respective labels
        """
        self.ifold = 0
        for train_index, test_index in self.cross_val.split(xdata[:, 0, 0], ydata):

            X_train_all, X_test_all = xdata[train_index], xdata[test_index]
            y_train, y_test = ydata[train_index].astype(
                int), ydata[test_index].astype(int)

            if return_idx:
                yield X_train_all, X_test_all, y_train, y_test, test_index
            else:
                yield X_train_all, X_test_all, y_train, y_test
            self.ifold += 1

    def roll_over_time(self, X_train_all, X_test_all=None):
        """
        returns one timepoint of EEG trial at a time
        """
        for self.itime, time in enumerate(self.t):
            time_window_idx = (self.samples >= time) & (
                self.samples < time + self.time_window)

            # Data for this time bin
            X_train = np.mean(X_train_all[..., time_window_idx], 2)
            if X_test_all is not None:
                X_test = np.mean(X_test_all[..., time_window_idx], 2)
                yield X_train, X_test
            else:
                yield X_train

    def roll_over_time_temp_gen(self, X_train_all, X_test_all):
        '''
        yield every other timepoint for each timepoint. Used for temporal generalizability plots.
        Not used in analyses for paper.
        '''

        for self.itime1, time1 in enumerate(self.t):
            for self.itime2, time2 in enumerate(self.t):
                time_window_idx1 = (self.samples >= time1) & (
                    self.samples < time1 + self.time_window)
                time_window_idx2 = (self.samples >= time2) & (
                    self.samples < time2 + self.time_window)

                # Data for this time bin
                X_train = np.mean(X_train_all[..., time_window_idx1], 2)
                X_test = np.mean(X_test_all[..., time_window_idx2], 2)

                yield X_train, X_test

    def train_test_custom_split(self, xdata_train, xdata_test, ydata_train, ydata_test):
        '''
        Takes in train and test data and yields portion of each for purposes of cross-validation.
        Useful if you want data to always be in train, and other data to always be in test.
        e.g. train on color and test on orientation data. 
        '''
        cross_val_test = StratifiedShuffleSplit(n_splits=1, test_size=.8)
        self.ifold = 0
        for train_index, _ in self.cross_val.split(xdata_train, ydata_train):
            X_train_all, y_train = xdata_train[train_index], ydata_train[train_index].astype(
                int)

            for _, test_index in cross_val_test.split(xdata_test, ydata_test):
                X_test_all, y_test = xdata_test[test_index], ydata_test[test_index].astype(
                    int)

            yield X_train_all, X_test_all, y_train, y_test
            self.ifold += 1


class Classification:
    def __init__(self, wrangl, nsub, num_labels=None, classifier=None):
        self.wrangl = wrangl
        self.n_splits = wrangl.n_splits
        self.t = wrangl.t
        if wrangl.num_labels:
            self.num_labels = wrangl.num_labels
        if num_labels:
            self.num_labels = num_labels
        if self.num_labels is None:
            raise Exception(
                'Must provide number of num_labels to Classification')

        self.nsub = nsub

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = LogisticRegression()
        self.scaler = StandardScaler()

        self.acc = np.zeros((self.nsub, np.size(self.t), self.n_splits))*np.nan
        self.acc_shuff = np.zeros(
            (self.nsub, np.size(self.t), self.n_splits))*np.nan
        self.conf_mat = np.zeros((self.nsub, np.size(
            self.t), self.n_splits, self.num_labels, self.num_labels))*np.nan

    def standardize(self, X_train, X_test):
        """
        z-score each electrode across trials at this time point

        returns standardized train and test data 
        Note: this fits and transforms train data, then transforms test data with mean and std of train data!!!
        """

        # Fit scaler to X_train and transform X_train
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test

    def decode(self, X_train, X_test, y_train, y_test, y_test_shuffle, isub):
        '''
        does actual training and testing of classifier after standardizing the data. Also does shuffled testing and confusion matrix.
        '''
        ifold = self.wrangl.ifold
        itime = self.wrangl.itime

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, itime, ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, itime, ifold] = self.classifier.score(
            X_test, y_test_shuffle)
        self.conf_mat[isub, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test))

    def decode_pairwise(self, X_train, X_test, y_train, y_test, y_test_shuffle, isub):
        '''
        Same functionality as decode. But results matrices are different shape.
        Used when using group_dict_list and rolling over multiple sets of classes (e.g. 1vs2 and 2vs4)
        '''

        ifold = self.wrangl.ifold
        itime = self.wrangl.itime
        iss = self.wrangl.iss

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, iss, itime, ifold] = self.classifier.score(
            X_test, y_test)
        self.acc_shuff[isub, iss, itime, ifold] = self.classifier.score(
            X_test, y_test_shuffle)
        self.conf_mat[isub, iss, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test))

    def decode_temp_gen(self, X_train, X_test, y_train, y_test, isub):
        '''
        Same functionality as decode. But results matrices are different shape.
        Not used in analyses for paper.
        '''
        ifold = self.wrangl.ifold
        itime1 = self.wrangl.itime1
        itime2 = self.wrangl.itime2

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, itime1, itime2,
                 ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, itime1, itime2, ifold] = self.classifier.score(
            X_test, np.random.permutation(y_test))
        self.conf_mat[isub, itime1, itime2, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test))

    def decode_electrode_subset(self, X_train, X_test, y_train, y_test, isub):
        '''
        Same functionality as decode. But results matrices are different shape.
        Not used in analyses for paper.
        '''

        ifold = self.wrangl.ifold
        itime = self.wrangl.itime
        ielec = self.wrangl.ielec

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, ielec, itime, ifold] = self.classifier.score(
            X_test, y_test)
        self.acc_shuff[isub, ielec, itime, ifold] = self.classifier.score(
            X_test, np.random.permutation(y_test))
        self.conf_mat[isub, ielec, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test))


class Interpreter:
    def __init__(
        self, clfr=None, subtitle='', output_dir=None, experiment_name=''
    ):
        if clfr is not None:
            self.clfr = clfr
            self.t = clfr.wrangl.t
            self.time_window = clfr.wrangl.time_window
            self.time_step = clfr.wrangl.time_step
            self.trial_average = clfr.wrangl.trial_average
            self.n_splits = clfr.wrangl.n_splits
            self.labels = list(clfr.wrangl.labels)
            self.electrodes = clfr.wrangl.electrodes
            self.acc = clfr.acc
            self.acc_shuff = clfr.acc_shuff
            self.conf_mat = clfr.conf_mat

        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = "Arial"
        matplotlib.rcParams['font.family'] = "sans-serif"

        self.timestr = time.strftime("%Y%m%d_%H%M")
        self.subtitle = subtitle
        self.experiment_name = experiment_name

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = Path('./output')
        self.fig_dir = self.output_dir / 'figures'

    def save_results(self,
                     additional_values=None
                     ):
        values = ['t', 'time_window', 'time_step', 'trial_average',
                  'n_splits', 'labels', 'electrodes', 'acc', 'acc_shuff', 'conf_mat']
        if additional_values:
            for val in additional_values:
                values.append(val)

        results_dict = {}
        for value in values:
            results_dict[value] = self.__dict__[value]

        filename = self.subtitle + 'bin' + \
            str(self.trial_average) + '_' + self.timestr + '.pickle'
        file_to_save = self.output_dir / filename

        with open(file_to_save, 'wb') as fp:
            pickle.dump(results_dict, fp)

    def load_results(self,
                     filename=None
                     ):

        if filename is None:
            list_of_files = self.output_dir.glob('*.pickle')
            file_to_open = max(list_of_files, key=os.path.getctime)
            print('No filename provided. Loading most recent results.')
        else:
            file_to_open = self.output_dir / filename

        with open(file_to_open, 'rb') as fp:
            results = pickle.load(fp)

        self.__dict__.update(results)

    def savefig(self, subtitle='', file_format=['.pdf', '.png'], save=True):
        if save:
            for file in file_format:
                filename = self.subtitle + subtitle + file
                output = self.fig_dir / filename
                plt.savefig(output, bbox_inches='tight',
                            dpi=1000, format=file[1:])
                print(f'Saving {output}')

    def plot_acc(self, subtitle='', significance_testing=False, stim_time=[0, 250],
                 savefig=False, title=None, ylim=[.18, .55], chance_text_y=.19):
        '''
        Plots classification accuracy
        '''
        acc = np.mean(self.acc, 2)
        se = sista.sem(acc, 0)
        acc_mean = np.mean(acc, 0)
        upper_bound, lower_bound = acc_mean + se, acc_mean - se
        acc_shuff = np.mean(self.acc_shuff, 2)
        se_shuff = sista.sem(acc_shuff, 0)
        acc_mean_shuff = np.median(acc_shuff, 0)
        upper_bound_shuff, lower_bound_shuff = acc_mean_shuff + \
            se_shuff, acc_mean_shuff - se_shuff
        chance = 1/len(self.labels)
        sig_y = chance-.05
        stim_lower = ylim[0]
        stim_upper = ylim[1]

        # plotting
        fig, ax = plt.subplots(1, figsize=(9, 6))

        ax.fill_between(stim_time, [stim_lower, stim_lower], [
                        stim_upper, stim_upper], color='gray', alpha=.5)
        ax.plot(self.t, np.ones((len(self.t)))*chance, '--', color='gray')
        ax.fill_between(self.t, upper_bound_shuff,
                        lower_bound_shuff, alpha=.5, color='gray')
        ax.plot(self.t, acc_mean_shuff, color='gray')
        ax.fill_between(self.t, upper_bound, lower_bound,
                        alpha=.5, color='tomato')
        ax.plot(self.t, acc_mean, color='tab:red', linewidth=2)

        # Significance Testing
        if significance_testing:
            p = np.zeros(len(self.t[self.t > 0]))
            # only test on timepoints after stimulus onset
            for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                # one-sided paired ttest
                _, p[i] = sista.ttest_rel(
                    a=acc[:, t], b=acc_shuff[:, t], alternative='greater')
            # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
            _, corrected_p, _, _ = multipletests(p, method='fdr_bh')
            sig05 = corrected_p < .05

            plt.scatter(self.t[self.t > 0][sig05], np.ones(sum(sig05))*(sig_y),
                        marker='s', s=45, c='tab:red', label='p < .05')

        # aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks(np.arange(.1, 1.1, .1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(-200, 1200)
        plt.ylim(ylim)
        plt.legend(loc='lower right', frameon=False, fontsize=11)

        # labelling
        plt.xlabel('Time from stimulus onset (ms)', fontsize=14)
        plt.ylabel('Classification accuracy', fontsize=14)
        ax.text(0.875, chance_text_y, 'Shuffle', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', color='grey')
        ax.text(0.2, .98, 'Stim', transform=ax.transAxes, fontsize=16,
                verticalalignment='top', color='white')
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig('acc'+subtitle, save=savefig)
        plt.show()

        sig_timepoints = self.t[self.t > 0][sig05]
        delay_period_acc = np.mean(acc_mean[self.t > 250])
        delay_period_sd = np.std(acc_mean[self.t > 250])

        print(f'Significant timepoints: {sig_timepoints}')
        print(f'Mean delay accuracy: {delay_period_acc}')
        print(f'Mean delay S.D.: {delay_period_sd}')

    def plot_acc_subset(self, subset_list, chance, sig_ys, subtitle='', significance_testing=False, stim_time=[0, 250],
                        savefig=False, title=None, ylim=[.18, .55], chance_text_y=.17):
        '''
        plots classification accuracies. Useful when doing classifcation on set size (1vs2,2vs3,3vs4) or electrode subsets (Frontal, Central, Parietal)
        '''

        # plotting
        ax = plt.subplot(111)
        stim_lower = ylim[0]+.01
        stim_upper = ylim[1]
        ax.fill_between(stim_time, [stim_lower, stim_lower], [
                        stim_upper, stim_upper], color='gray', alpha=.5)
        ax.plot(self.t, np.ones((len(self.t)))*chance, '--', color='gray')
        colors = ['royalblue', 'firebrick', 'forestgreen', 'orange', 'purple']

        # sig_ys = [.2125,.1875,.2]
        for isubset, subset in enumerate(subset_list):
            color = colors[isubset]
            acc = self.acc[:, isubset]
            acc_shuff = self.acc_shuff[:, isubset]

            acc = np.mean(acc, 2)
            se = sista.sem(acc, 0)
            acc_mean = np.mean(acc, 0)
            upper_bound, lower_bound = acc_mean + se, acc_mean - se
            acc_shuff = np.mean(acc_shuff, 2)
            se_shuff = sista.sem(acc_shuff, 0)
            acc_mean_shuff = np.mean(acc_shuff, 0)
            upper_bound_shuff, lower_bound_shuff = acc_mean_shuff + \
                se_shuff, acc_mean_shuff - se_shuff

            ax.fill_between(self.t, upper_bound_shuff,
                            lower_bound_shuff, alpha=.2, color='gray')
            ax.plot(self.t, acc_mean_shuff, color='gray')
            ax.fill_between(self.t, upper_bound, lower_bound,
                            alpha=.5, color=color)
            ax.plot(self.t, acc_mean, color=color, label=subset)

            # Significance Testing
            if significance_testing:
                p = np.zeros(len(self.t[self.t > 0]))
                # only test on timepoints after stimulus onset
                for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                    # one-sided paired ttest
                    _, p[i] = sista.ttest_rel(
                        a=acc[:, t], b=acc_shuff[:, t], alternative='greater')

                # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
                _, corrected_p, _, _ = multipletests(p, method='fdr_bh')
                sig05 = corrected_p < .05

                plt.scatter(self.t[self.t > 0][sig05]-10, np.ones(sum(sig05))*(sig_ys[isubset]),
                            marker='s', s=28, c=color)

            sig_timepoints = self.t[self.t > 0][sig05]
            delay_period_acc = np.mean(acc_mean[self.t > 250])
            delay_period_sd = np.std(acc_mean[self.t > 250])
            print(f'{subset} significant timepoints: {sig_timepoints}')
            print(f'{subset} mean delay accuracy: {delay_period_acc}')
            print(f'{subset} mean delay S.D.: {delay_period_sd}')

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(handles, subset_list,
                        title='Classification', fontsize=12)
        plt.setp(leg.get_title(), fontsize=12)

        # aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks(np.arange(.1, 1.1, .1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(-200, 1200)
        plt.ylim(ylim)

        # labelling
        plt.xlabel('Time from stimulus onset (ms)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        ax.text(.88, chance_text_y, 'Shuffle', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', color='grey')
        ax.text(0.185, .98, 'Stim', transform=ax.transAxes, fontsize=16,
                verticalalignment='top', color='white')
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig('acc'+subtitle, save=savefig)
        plt.show()

    def plot_acc_compare_subset(self, subset_list, chance, subtitle='', significance_testing=False, stim_time=[0, 250],
                                savefig=False, title=None, ylim=[.18, .55], chance_text_y=.17, legend_title=''):
        '''
        plots classification accuracies. Useful when doing classifcation on set size (1vs2,2vs3,3vs4) or electrode subsets (Frontal, Central, Parietal)
        '''

        # plotting
        fig, ax = plt.subplots(1, figsize=(9, 6))
        stim_lower = ylim[0]
        stim_upper = ylim[1]
        ax.fill_between(stim_time, [stim_lower, stim_lower], [
                        stim_upper, stim_upper], color='gray', alpha=.5)
        ax.plot(self.t, np.ones((len(self.t)))*chance, '--', color='gray')
        colors = ['royalblue', 'firebrick', 'forestgreen', 'orange', 'purple']
        sig_y = chance-.05

        for isubset, subset in enumerate(subset_list):
            color = colors[isubset]
            acc = self.acc[:, isubset]
            acc_shuff = self.acc_shuff[:, isubset]

            acc = np.mean(acc, 2)
            se = sista.sem(acc, 0)
            acc_mean = np.mean(acc, 0)
            upper_bound, lower_bound = acc_mean + se, acc_mean - se
            acc_shuff = np.mean(acc_shuff, 2)
            se_shuff = sista.sem(acc_shuff, 0)
            acc_mean_shuff = np.mean(acc_shuff, 0)
            upper_bound_shuff, lower_bound_shuff = acc_mean_shuff + \
                se_shuff, acc_mean_shuff - se_shuff

            if isubset == 1:
                ax.fill_between(self.t, upper_bound_shuff,
                                lower_bound_shuff, alpha=.2, color='gray')
                ax.plot(self.t, acc_mean_shuff, color='gray')
            ax.fill_between(self.t, upper_bound, lower_bound,
                            alpha=.5, color=color)
            ax.plot(self.t, acc_mean, color=color, label=subset)

            delay_period_acc = np.mean(acc_mean[self.t > 250])
            delay_period_sd = np.std(acc_mean[self.t > 250])
            print(f'{subset} mean delay accuracy: {delay_period_acc}')
            print(f'{subset} mean delay S.D.: {delay_period_sd}')

        # Significance Testing
        if significance_testing:
            acc1 = np.mean(self.acc[:, 0], 2)
            acc2 = np.mean(self.acc[:, 1], 2)

            p = np.zeros(len(self.t[self.t > 0]))
            # only test on timepoints after stimulus onset
            for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                # one-sided paired ttest
                _, p[i] = sista.ttest_rel(
                    a=acc1[:, t], b=acc2[:, t], alternative='greater')

            # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
            _, corrected_p, _, _ = multipletests(p, method='fdr_bh')
            sig05 = corrected_p < .05

            plt.scatter(self.t[self.t > 0][sig05]-10, np.ones(sum(sig05))*(sig_y),
                        marker='s', s=28, c='purple')

        sig_timepoints = self.t[self.t > 0][sig05]
        delay_period_acc = np.mean(acc_mean[self.t > 250])
        delay_period_sd = np.std(acc_mean[self.t > 250])
        print(f'significant timepoints: {sig_timepoints}')
        print(f'mean delay accuracy: {delay_period_acc}')
        print(f'mean delay S.D.: {delay_period_sd}')

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(handles, subset_list, title=legend_title, fontsize=12)
        plt.setp(leg.get_title(), fontsize=12)

        # aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks(np.arange(.1, 1.1, .1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(-200, 1200)
        plt.ylim(ylim)

        # labelling
        plt.xlabel('Time from stimulus onset (ms)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        ax.text(.88, chance_text_y, 'Shuffle', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', color='grey')
        ax.text(0.2, .98, 'Stim', transform=ax.transAxes, fontsize=16,
                verticalalignment='top', color='white')
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig('acc'+subtitle, save=savefig)
        plt.show()

    def plot_conf_mat(self, subtitle='', color_map=plt.cm.RdGy_r, lower=0, upper=1, savefig=False, subplot=111):
        """
        plots the confusion matrix for the classifier

        Input:
        self.conf_mat of shape [subjects,timepoints,folds,setsizeA,setsizeB]
        """
        cm = np.mean(np.mean(np.mean(self.conf_mat[:, self.t > 250], 2), 1), 0)

        # Normalize
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Generate plot
        ax = plt.subplot(subplot)

        plt.imshow(cm, interpolation='nearest',
                   cmap=color_map, clim=(lower, upper))

        # for font color readability
        thresh = np.percentile([lower, upper], [25, 75])
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i, j] > thresh[1] or cm[i, j] < thresh[0]:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color=color,
                    fontsize=16)

        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels)
        plt.yticks(tick_marks, self.labels)
        plt.ylabel('True Set Size', fontsize=14)
        plt.xlabel('Predicted Set Size', fontsize=14)
        plt.tight_layout()
        plt.gca().invert_yaxis()

        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        self.savefig('conf_mat'+subtitle, save=savefig)
        plt.show()

    def temporal_generalizability(self, cmap=plt.cm.viridis, lower_lim=0, upper_lim=1, savefig=False):
        """
        Plot temporal generalizability
        Not used in analyses for paper.
        """

        plt.figure()
        plt.imshow(np.mean(np.mean(self.acc, 0), 2),
                   interpolation='nearest', cmap=cmap, clim=(lower_lim, upper_lim))
        # plt.title('Accuracy for Training/Testing\non Different Timepoints')
        plt.colorbar()

        tick_marks = np.arange(0, len(self.t), 4)

        plt.xticks(tick_marks, self.t[0::4])
        plt.yticks(tick_marks, self.t[0::4])

        plt.xlabel('Testing Timepoint (ms)')
        plt.ylabel('Training Timepoint (ms)')
        plt.gca().invert_yaxis()

        self.savefig('temp_gen_', save=savefig)

    # FIX THIS
    def plot_acc_single_feature(self, other_acc, other_acc_shuff, subtitle='', significance_testing=False, stim_time=[0, 250],
                                savefig=False, title=False, lower=.185, upper=.55, chance_text_y=.17):

        acc_feat1 = np.mean(self.acc, 2)
        acc_feat2 = np.mean(other_acc, 2)

        # two gray lines
        acc_feat1_mean = np.mean(acc_feat1, 0)
        acc_feat2_mean = np.mean(acc_feat2, 0)

        # single feature
        acc_single_feature_subs = (acc_feat1 + acc_feat2)/2
        acc_single_feature_subs_se = sista.sem(acc_single_feature_subs, 0)
        acc_single_feature_mean = (acc_feat1_mean + acc_feat2_mean)/2

        acc_shuff = np.mean(self.acc_shuff, 2)
        acc_shuff2 = np.mean(other_acc_shuff, 2)
        acc_shuff_both = (acc_shuff+acc_shuff2)/2
        acc_shuff = np.mean(acc_shuff_both, 0)
        se_shuff = sista.sem(acc_shuff_both, 0)

        upper_bound, lower_bound = acc_single_feature_mean + \
            acc_single_feature_subs_se, acc_single_feature_mean - acc_single_feature_subs_se
        upper_bound_shuff, lower_bound_shuff = acc_shuff + se_shuff, acc_shuff - se_shuff

        chance = 1/len(self.labels)

        # plotting
        ax = plt.subplot(111)

        ax.fill_between(stim_time, [lower, lower], [
                        upper, upper], color='gray', alpha=.5)
        ax.plot(self.t, np.ones((len(self.t)))*chance, '--', color='gray')
        ax.fill_between(self.t, upper_bound_shuff,
                        lower_bound_shuff, alpha=.5, color='gray')
        ax.plot(self.t, acc_shuff, color='gray')
        ax.fill_between(self.t, upper_bound, lower_bound,
                        alpha=.5, color='tomato')
        # ax.plot(self.t,acc_feat1_median,c='gray',linewidth=2,alpha=.5)
        # ax.plot(self.t,acc_feat2_median,c='gray',linewidth=2,alpha=.5)
        ax.plot(self.t, acc_single_feature_mean, color='tab:red', linewidth=2)

        # # Significance Testing
        if significance_testing:
            p = np.zeros((self.t.shape[0]))
            for i in range(len(self.t)):
                # wilcoxon is non-parametric paired ttest basically
                _, p[i] = sista.wilcoxon(
                    x=acc_single_feature_subs[:, i], y=acc_shuff_both[:, i], alternative='greater')

            # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
            _, corrected_p, _, _ = multipletests(p, method='fdr_bh')
            sig05 = corrected_p < .05

            plt.scatter(self.t[sig05]+10, np.ones(sum(sig05))*(.2),
                        marker='s', s=25, c='tab:red')

        # aesthetics
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks(np.arange(.2, .6, .1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(-200, 1250)
        plt.ylim(.18, .55)

        # labelling
        plt.xlabel('Time from stimulus onset (ms)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        ax.text(0.85, chance_text_y, 'Chance', transform=ax.transAxes, fontsize=13,
                verticalalignment='top', color='grey')
        ax.text(0.188, .98, 'Stim', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', color='white')

        self.savefig('acc'+subtitle, save=savefig)
        plt.show()


class ERP:

    def __init__(self, exp, subtitle='', fig_dir=None):
        self.exp = exp
        self.info = exp.info
        self.xdata_files = exp.xdata_files
        self.ydata_files = exp.ydata_files

        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.subtitle = subtitle

        if fig_dir:
            self.fig_dir = fig_dir
        else:
            self.fig_dir = Path('output/figures')

    def savefig(self, subtitle='', file_format='.pdf', save=True):
        if save:
            filename = self.subtitle + subtitle + self.timestr + file_format
            output = self.fig_dir / filename
            plt.savefig(output, bbox_inches='tight',
                        dpi=1000, format=file_format[1:])
            print(f'Saving {output}')

    def load_all_eeg(self):

        xdata_all = np.empty((self.exp.nsub), dtype='object')
        ydata_all = np.empty((self.exp.nsub), dtype='object')
        for isub in range(self.exp.nsub):
            xdata_all[isub], ydata_all[isub] = self.exp.load_eeg(isub)
        return xdata_all, ydata_all

    def _select_electrodes(self, xdata, electrode_subset=None, electrode_idx=None):

        if electrode_subset is not None:
            # Create index for electrodes to include in plot
            electrode_labels = [el for n, el in enumerate(
                self.info['chan_labels']) if el.startswith(electrode_subset)]
            electrode_idx = np.in1d(self.info['chan_labels'], electrode_labels)
            xdata = xdata[electrode_idx]
        elif electrode_idx is not None:
            xdata = xdata[electrode_idx]

        return xdata

    def plot_ss(self, xdata_all, ydata_all, subtitle='',
                electrode_subset=None, electrode_idx=None,
                savefig=False):

        ss_data = np.zeros((self.exp.nsub, len(
            np.unique(ydata_all[0])), len(self.info['times'])))  # hardcode
        for isub in range(self.exp.nsub):
            xdata = xdata_all[isub]
            ydata = ydata_all[isub]
            for iss, ss in enumerate(np.unique(ydata_all[0])):
                ss_idx = ydata == ss
                data = np.mean(xdata[ss_idx], 0)
                ss_data[isub, iss] = np.mean(self._select_electrodes(
                    data, electrode_subset, electrode_idx), 0)

        ax = plt.subplot(111)

        upper, lower = 0, 0
        for iss, ss in enumerate(np.unique(ydata_all[0])):
            x = np.mean(ss_data[:, iss], 0)
            se = np.std(ss_data[:, iss], 0)/np.sqrt(self.exp.nsub)

            # ERP
            plt.plot(self.info['times'], x,
                     label=f'{ss}', linewidth=2.5, alpha=0.8)
            # SE
            plt.fill_between(self.info['times'], x-se, x+se, alpha=.3)

            maxi, mini = max(x) + max(se), min(x) - min(se)
            print(maxi)
            if maxi > upper:
                upper = maxi
            if mini < lower:
                lower = mini

        # Grey stim bar
        plt.fill_between([0, 250], [lower-1, lower-1],
                         [upper+.45, upper+.45], color='gray', alpha=.5, zorder=0)

        # Hide the right and top spines]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Cleaning up plot
        plt.gca().invert_yaxis()
        plt.legend(title='Set Size', loc='lower right')
        plt.xlabel('Time from Array Onset (ms)')
        plt.ylabel("Amplitude (microvolts)")

        self.savefig(subtitle=subtitle, save=savefig)
        plt.show()

    def plot_feat(self, xdata_all, ydata_all, subtitle='',
                  electrode_subset=None, electrode_idx=None,
                  savefig=False):
        # first dimension of xdata_all should be exp

        mean_exp = np.empty((len(xdata_all), len(self.info['times'])))
        se_exp = np.empty((len(xdata_all), len(self.info['times'])))

        for iexp in range(len(xdata_all)):
            xdata_exp = xdata_all[iexp]
            ydata_exp = ydata_all[iexp]
            xdata_subs = np.empty((len(xdata_exp), len(self.info['times'])))
            for isub in range(len(xdata_exp)):
                xdata, ydata = xdata_exp[isub], ydata_exp[isub]
                xdata = np.mean(xdata[ydata == 1], 0)
                xdata_subs[isub] = np.mean(
                    self._select_electrodes(xdata, ('P', 'O', 'C')), 0)

            mean_exp[iexp] = np.mean(xdata_subs, 0)
            se_exp[iexp] = np.std(xdata_subs, 0)/np.sqrt(xdata_subs.shape[0])

        ax = plt.subplot(111)
        feats = ['Color', 'Orientation', 'Conjunction']
        for iexp, exp in enumerate(feats):
            x = mean_exp[iexp]
            se = se_exp[iexp]

            # ERP
            plt.plot(self.info['times'], x,
                     label=f'{exp}', linewidth=2.5, alpha=0.8)
            # SE
            plt.fill_between(self.info['times'], x-se, x+se, alpha=.3)

        # Grey stim bar
        upper = 6
        lower = -4
        plt.fill_between([0, 250], [upper, upper], [
                         lower, lower], color='gray', alpha=.1)

        # Hide the right and top spines]
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Cleaning up plot
        plt.gca().invert_yaxis()
        plt.legend(title='Feature', loc='lower right')
        plt.xlabel('Time from Array Onset (ms)')
        plt.ylabel("Amplitude (microvolts)")

        self.savefig(subtitle=subtitle, save=savefig)
        plt.show()
