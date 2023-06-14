from pathlib import Path
import scipy.io as sio
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from copy import deepcopy
import scipy.stats as sista

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.multitest import multipletests


class Experiment:
    def __init__(
        self,
        experiment_name,
        data_dir,
        info_from_file=True,
        dev=False,
        info_variable_names=[
            "unique_id",
            "chan_labels",
            "chan_x",
            "chan_y",
            "chan_z",
            "sampling_rate",
            "times",
        ],
        trim_timepoints=None,
    ):
        """Organizes and loads in EEG, trial labels, behavior, eyetracking, and session data.

        Keyword arguments:
        experiment_name -- name of experiment
        data_dir -- directory of data files
        info_from_file -- pull info from 0th info file in data_dir (default True)
        dev -- development mode: only use first 3 subjects' data (default False)
        info_variable_names -- names of variables to pull from info file
        trim_timepoints -- trims info.times and all loaded EEG data
        """
        self.experiment_name = experiment_name
        self.data_dir = Path(data_dir)
        self.trim_idx = None

        self.xdata_files = sorted(list(self.data_dir.glob("*xdata*.mat")))
        self.ydata_files = sorted(list(self.data_dir.glob("*ydata*.mat")))
        if dev:
            self.xdata_files = self.xdata_files[0:3]
            self.ydata_files = self.ydata_files[0:3]
        self.nsub = len(self.xdata_files)

        self.behavior_files = None
        self.artifact_idx_files = None
        self.info_files = None

        if info_from_file:
            self.info = self.load_info(0, info_variable_names)
            self.info.pop("unique_id")

            if trim_timepoints:
                self.trim_idx = (self.info["times"] >= trim_timepoints[0]) & (
                    self.info["times"] <= trim_timepoints[1]
                )
                self.info["original_times"] = self.info["times"]
                self.info["times"] = self.info["times"][self.trim_idx]

    def load_eeg(self, isub):
        """
        loads xdata (eeg data) and ydata (trial labels) from .mat

        Keyword arguments:
        isub -- index of subject to load
        """
        subj_mat = sio.loadmat(self.xdata_files[isub], variable_names=["xdata"])
        xdata = np.moveaxis(subj_mat["xdata"], [0, 1, 2], [1, 2, 0])

        if self.trim_idx is not None:
            xdata = xdata[:, :, self.trim_idx]

        ydata = self.load_ydata(isub)

        return xdata, ydata

    def load_ydata(self, isub):
        """
        loads ydata (trial labels) from .mat

        Keyword arguments:
        isub -- index of subject to load
        """
        subj_mat = sio.loadmat(self.ydata_files[isub], variable_names=["ydata"])
        ydata = np.squeeze(subj_mat["ydata"])

        return ydata

    def load_behavior(self, isub, remove_artifact_trials=True):
        """
        returns behavior from csv as dictionary

        Keyword arguments:
        isub -- index of subject to load
        remove_artifact_trials -- remove all behavior trials that were excluded from EEG data due to artifacts
        """
        if self.behavior_files is None:
            self.behavior_files = sorted(list(self.data_dir.glob("*behavior.csv")))

        behavior = pd.read_csv(self.behavior_files[isub]).to_dict("list")

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
        returns artifact index from EEG artifact rejection. useful for removing behavior trials not included in EEG data.

        Keyword arguments:
        isub -- index of subject to load
        """

        if not self.artifact_idx_files:
            self.artifact_idx_files = sorted(list(self.data_dir.glob("*artifact_idx*.mat")))

        artifact_idx = np.squeeze(sio.loadmat(self.artifact_idx_files[isub])["artifact_idx"] == 1)

        return artifact_idx

    def load_info(
        self,
        isub,
        variable_names=[
            "unique_id",
            "chan_labels",
            "chan_x",
            "chan_y",
            "chan_z",
            "sampling_rate",
            "times",
        ],
    ):
        """
        loads info file that contains data about EEG file and subject

        Keyword arguments:
        isub -- index of subject to load
        variable_names -- names of variables to pull from info file
        """
        if not self.info_files:
            self.info_files = sorted(list(self.data_dir.glob("*info*.mat")))

        info_file = sio.loadmat(self.info_files[isub], variable_names=variable_names)

        info = {k: np.squeeze(info_file[k]) for k in variable_names}

        return info


class Experiment_Syncer:
    def __init__(self, experiments, wrangler, train_group, get_matched_data=True):
        """
        Synchronizes subject data across multiple experiments.

        Keyword variables:
        experiments -- Experiments objects to be synced
        wrangler -- Wrangler object to be used
        train_group -- which experiments to be used in the training set
        get_matched_data -- only use subjects who appear in both experiments (default True)
        """
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
        """
        Loads all IDs in all experiments
        """
        self.all_ids = []
        for exp in self.experiments:
            exp.unique_ids = []
            for isub in range(exp.nsub):
                exp.unique_ids.append(int(exp.load_info(isub)["unique_id"]))
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
        """
        Finds only IDs that are in all experiments
        """
        self.id_dict = dict.fromkeys(self.matched_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.matched_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(m)
                except ValueError:
                    pass
        self.nsub = len(self.matched_ids)

    def _find_all_ids(self):
        """
        Finds IDs in all experiments. Used for loading all data across experiments.
        """
        self.id_dict = dict.fromkeys(self.all_ids)
        for k in self.id_dict.keys():
            self.id_dict[k] = dict.fromkeys(self.experiment_names)

        for exp in self.experiments:
            for m in self.all_ids:
                try:
                    self.id_dict[m][exp.experiment_name] = exp.unique_ids.index(m)
                except ValueError:
                    pass
        self.nsub = len(self.all_ids)

    def load_eeg(self, sub):
        """
        loads xdata (eeg data) and ydata (trial labels) of every experiment from .mat

        Keyword arguments:
        sub -- unique ID of subject to load
        """

        xdata = dict.fromkeys(self.experiment_names)
        ydata = dict.fromkeys(self.experiment_names)
        for exp in self.experiments:
            if self.id_dict[sub][exp.experiment_name] is not None:
                xdata[exp.experiment_name], ydata[exp.experiment_name] = exp.load_eeg(
                    self.id_dict[sub][exp.experiment_name]
                )
            else:
                xdata.pop(exp.experiment_name)
                ydata.pop(exp.experiment_name)
        return xdata, ydata

    def load_behavior(self, sub):
        """
        returns behavior from csv as dictionary

        Keyword arguments:
        sub -- unique ID of subject to load
        """
        beh = dict.fromkeys(self.experiment_names)
        for exp in self.experiments:
            if self.id_dict[sub][exp.experiment_name] is not None:
                beh[exp.experiment_name] = exp.load_behavior(self.id_dict[sub][exp.experiment_name])
            else:
                beh.pop(exp.experiment_name)
        return beh

    def select_labels(self, xdata, ydata):
        """
        includes labels only wanted for decoding. returns xdata and ydata with unwanted labels removed.

        Keyword arguments:
        xdata: eeg data, shape[electrodes,timepoints,trials]
        ydata: labels, shape[trials]
        """
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.select_labels(
                xdata[exp_name], ydata[exp_name]
            )
        return xdata, ydata

    def group_labels(self, xdata, ydata):
        """
        groups classes based on group_dict, removes not-included classes

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        """
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.group_labels(
                xdata[exp_name], ydata[exp_name]
            )
        return xdata, ydata

    def balance_labels(self, xdata, ydata):
        """
        balances number of class instances

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        """
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.balance_labels(
                xdata[exp_name], ydata[exp_name]
            )
        return xdata, ydata

    def bin_trials(self, xdata, ydata):
        """
        bins trials based on trial_bin_size

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        """
        for exp_name in xdata.keys():
            xdata[exp_name], ydata[exp_name] = self.wrangler.average_trials(
                xdata[exp_name], ydata[exp_name]
            )
        return xdata, ydata

    def setup_data(self, xdata, ydata, labels=False, group_dict=False):
        """
        does basic data manipulation using other functions. Deprecated.

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        labels -- use select_labels function (default False)
        group_dict -- use group.labels function (default False)
        """
        if labels:
            xdata, ydata = self.select_labels(xdata, ydata)
        if group_dict:
            xdata, ydata = self.group_labels(xdata, ydata)
        xdata, ydata = self.balance_labels(xdata, ydata)
        xdata, ydata = self.bin_trials(xdata, ydata)
        return xdata, ydata

    def pairwise(self, xdata_all, ydata_all):
        """
        When using group_dict_list (e.g. 1vs2 then 2vs4), yields data with only those classes.

        Keyword arguments:
        xdata_all -- eeg data, shape[electrodes,timepoints,trials]
        ydata_all -- labels, shape[trials]
        """

        for self.wrangler.iss, ss in enumerate(self.wrangler.group_dict_list):
            xdata, ydata = deepcopy(xdata_all), deepcopy(ydata_all)

            self.wrangler.group_dict = ss

            for exp_name in xdata.keys():
                xdata[exp_name], ydata[exp_name] = self.wrangler.group_labels(
                    xdata[exp_name], ydata[exp_name]
                )
            yield xdata, ydata

    def group_data(self, xdata, ydata):
        """
        groups data into train and test groups based on self.train_group

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        """
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
        if (
            xdata_test is None
        ):  # if both groups are in train_group, function combines and returns as one
            return xdata_train, ydata_train
        else:
            return xdata_train, xdata_test, ydata_train, ydata_test


class Wrangler:
    def __init__(
        self,
        samples,
        time_window,
        time_step,
        trial_bin_size,
        n_splits,
        group_dict=None,
        group_dict_list=None,
        train_labels=None,
        test_size=0.1,
        labels=None,
        electrodes=None,
        electrode_subset_list=None,
    ):
        """
        Handles data processing and cross-validation.

        Keyword arguments:
        samples -- timepoints (in ms) of EEG epochs
        time_window -- window size for averaging
        time_step -- window step for averaging
        trial_bin_size -- number of trials per trial bin
        n_splits -- number of folds in cross-validation procedure
        group_dict -- trial labels to be grouped together (default None)
        group_dict_list -- list of group_dict for pairwise decoding (default None)
        train_labels -- list of labels to include in training (default None)
        test_size -- percent of trials to test (default 0.1)
        labels -- labels to be included in decoding (default None)
        electrodes -- names of electrodes in EEG data (default None)
        electrode_subset_list -- which electrodes to include in decoding (default None)
        """
        self.samples = samples
        self.sample_step = samples[1] - samples[0]
        self.time_window = time_window
        self.time_step = time_step
        self.trial_bin_size = trial_bin_size
        self.n_splits = n_splits
        self.test_size = test_size
        self.group_dict = group_dict
        self.group_dict_list = group_dict_list
        self.train_labels = train_labels
        self.labels = labels
        self.electrodes = electrodes
        self.electrode_subset_list = electrode_subset_list

        if self.group_dict_list:
            self.labels = []
            self.label_dict = []
            self.num_labels = []
            for group_dict in group_dict_list:
                labels = list(group_dict)
                self.labels.append(labels)
                label_dict = {}
                for i, key in enumerate(group_dict.keys()):
                    label_dict[key] = i
                self.label_dict.append(label_dict)
                self.num_labels.append(len(labels))
        else:
            if self.group_dict:
                self.labels = list(self.group_dict.keys())
                self.label_dict = {}
                for i, key in enumerate(group_dict.keys()):
                    self.label_dict[key] = i
            if self.labels:
                self.num_labels = len(self.labels)
            else:
                self.num_labels = None

        self.t = samples[
            0 : samples.shape[0]
            - int(time_window / self.sample_step)
            + 1 : int(time_step / self.sample_step)
        ]

    def select_labels(self, xdata, ydata, labels=None, return_idx=False):
        """
        includes labels only wanted for decoding. returns xdata and ydata with unwanted labels removed.

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        labels -- list of labels to include
        return_idx -- return index of trials selected
        """

        if labels is None:
            labels = self.labels

        label_idx = np.isin(ydata, labels)
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

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        empty_val -- pre-allocate empty array with this value.
        """

        xdata_new = np.ones(xdata.shape) * empty_val
        ydata_new = np.ones(ydata.shape) * empty_val

        for i, k in enumerate(self.group_dict.values()):
            trial_idx = np.arange(ydata.shape[0])[np.isin(ydata, k)]
            xdata_new[trial_idx] = xdata[trial_idx]
            ydata_new[trial_idx] = i

        trial_idx = ydata_new == empty_val
        return xdata_new[~trial_idx], ydata_new[~trial_idx]

    def pairwise(self, xdata_all, ydata_all):
        """
        When using group_dict_list (e.g. 1vs2 then 2vs4), yields data with only those classes.

        Keyword arguments:
        xdata_all -- eeg data, shape[electrodes,timepoints,trials]
        ydata_all -- labels, shape[trials]
        """
        for self.iss, ss in enumerate(self.group_dict_list):
            xdata, ydata = deepcopy(xdata_all), deepcopy(ydata_all)
            self.group_dict = ss
            yield self.group_labels(xdata, ydata)

    def balance_labels(self, xdata, ydata, downsamp=None):
        """
        balances number of class instances

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        downsamp -- number of trials to downsample to (default None). If None, downsamples to lowest count.
        """
        unique_labels, counts_labels = np.unique(ydata, return_counts=True)
        if downsamp is None:
            downsamp = min(counts_labels)
        label_idx = []
        for label in unique_labels:
            label_idx = np.append(
                label_idx,
                np.random.choice(np.arange(len(ydata))[ydata == label], downsamp, replace=False),
            )

        xdata = xdata[label_idx.astype(int), :, :]
        ydata = ydata[label_idx.astype(int)]

        return xdata, ydata

    def bin_trials(self, xdata, ydata, permute_trials=True):
        """
        bins trials based on trial_bin_size

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        permute_trials -- shuffle trials before binning to get unique bins each call
        """
        if self.trial_bin_size:
            if permute_trials:
                p = np.random.permutation(len(ydata))
                xdata, ydata = xdata[p], ydata[p]

            # get labels and counts
            unique_labels, label_counts = np.unique(ydata, return_counts=True)

            # determine number of bins per label
            n_bins = label_counts // self.trial_bin_size
            n_trials = n_bins * self.trial_bin_size

            xdata_bin = []
            ydata_bin = []
            # loop through labels
            for ilabel, label in enumerate(unique_labels):
                # assign each trial of label to bin
                label_bins = np.tile(np.arange(n_bins[ilabel]), n_trials[ilabel] // n_bins[ilabel])
                # create label index
                label_idx = ydata == label
                # grab data
                label_data = xdata[label_idx][: n_trials[ilabel]]

                # preallocate
                bin_average_data = np.empty(
                    (n_bins[ilabel], label_data.shape[1], label_data.shape[2])
                )
                # loop though bins
                for ibin, bin in enumerate(np.unique(label_bins)):
                    # make bin idx
                    bin_idx = label_bins == bin
                    # average over data
                    bin_average_data[ibin] = np.mean(label_data[bin_idx], 0)

                xdata_bin.append(bin_average_data)
                ydata_bin += [label] * n_bins[ilabel]

            xdata_bin = np.concatenate(xdata_bin)
            ydata_bin = np.array(ydata_bin)
            return xdata_bin, ydata_bin
        else:
            return xdata, ydata

    def bin_data(self, X_train_all, X_test_all, y_train, y_test):
        """
        helper function than does trial binning

        Keyword arguments:
        X_train_all -- EEG data to be used for training
        X_test_all -- EEG data to be used for testing
        y_train -- trial labels for training data
        y_test -- trial labels for testing data
        """

        X_train_all, y_train = self.bin_trials(X_train_all, y_train)
        X_test_all, y_test = self.bin_trials(X_test_all, y_test)

        return X_train_all, X_test_all, y_train, y_test

    def balance_data(self, X_train_all, X_test_all, y_train, y_test):
        """
        helper function than does trial binning and balances data

        Keyword arguments:
        X_train_all -- EEG data to be used for training
        X_test_all -- EEG data to be used for testing
        y_train -- trial labels for training data
        y_test -- trial labels for testing data
        """

        X_train_all, y_train = self.balance_labels(X_train_all, y_train)
        X_test_all, y_test = self.balance_labels(X_test_all, y_test)

        return X_train_all, X_test_all, y_train, y_test

    def bin_and_balance_data(self, X_train_all, X_test_all, y_train, y_test):
        """
        helper function than does trial binning and balances data

        Keyword arguments:
        X_train_all -- EEG data to be used for training
        X_test_all -- EEG data to be used for testing
        y_train -- trial labels for training data
        y_test -- trial labels for testing data
        """

        X_train_all, X_test_all, y_train, y_test = self.bin_data(
            X_train_all, X_test_all, y_train, y_test
        )
        X_train_all, X_test_all, y_train, y_test = self.balance_data(
            X_train_all, X_test_all, y_train, y_test
        )

        return X_train_all, X_test_all, y_train, y_test

    def select_training_data(self, X_train_all, y_train):
        """
        select training data based on self.train_labels

        Keyword arguments:
        X_train_all -- EEG data to be used for training
        y_train -- trial labels for training data
        """

        # create index for labels from train_labels
        labels = []
        [labels.append(self.label_dict[k]) for k in self.train_labels]
        return self.select_labels(X_train_all, y_train, labels)

    def select_electrodes(self, xdata, electrode_subset=None):
        """
        removes electrodes not included in electrode_subset.

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        electrode_subset -- electrode subset to select for use in classification (default None)
        """
        if electrode_subset is not None:
            # Create index for electrodes to include in plot
            electrode_labels = [
                el for n, el in enumerate(self.electrodes) if el.startswith(electrode_subset)
            ]
            electrode_idx = np.in1d(self.electrodes, electrode_labels)

            xdata = xdata[:, electrode_idx]

        return xdata

    def roll_over_electrodes(self, xdata_all, ydata_all):
        """
        yields data with electrodes not in electrode subset, iterating over electrode_subset_list.

        Keyword arguments:
        xdata_all -- eeg data, shape[electrodes,timepoints,trials]
        ydata_all -- labels, shape[trials]
        """

        for self.ielec, electrode_subset in enumerate(self.electrode_subset_list):
            yield self.select_electrodes(xdata_all, electrode_subset), ydata_all

    def bin_and_split_data(self, xdata, ydata, test_size=0.20):
        """
        returns xtrain and xtest data and labels, binned

        Keyword arguments:
        xdata -- eeg data, shape[electrodes,timepoints,trials]
        ydata -- labels, shape[trials]
        return_idx -- return index used to select test data (default False)
        """

        for self.ifold in range(self.n_splits):
            xdata_binned, ydata_binned = self.bin_trials(xdata, ydata)
            X_train_all, X_test_all, y_train, y_test = train_test_split(
                xdata_binned, ydata_binned, stratify=ydata_binned, test_size=test_size
            )

            yield X_train_all, X_test_all, y_train, y_test

    def roll_over_time(self, X_train_all, X_test_all=None):
        """
        returns one timepoint of EEG trial at a time

        Keyword arguments:
        X_train_all -- all EEG data for training
        X_test_all -- all EEG data for testing (default None)
        """
        for self.itime, time in enumerate(self.t):
            time_window_idx = (self.samples >= time) & (self.samples < time + self.time_window)

            # Data for this time bin
            X_train = np.mean(X_train_all[..., time_window_idx], 2)
            if X_test_all is not None:
                X_test = np.mean(X_test_all[..., time_window_idx], 2)
                yield X_train, X_test
            else:
                yield X_train

    def roll_over_time_temp_gen(self, X_train_all, X_test_all):
        """
        yield every other timepoint for each timepoint. Used for temporal generalizability plots.

        Keyword arguments:
        X_train_all -- all EEG data for training
        X_test_all -- all EEG data for testing (default None)
        """

        for self.itime1, time1 in enumerate(self.t):
            for self.itime2, time2 in enumerate(self.t):
                time_window_idx1 = (self.samples >= time1) & (
                    self.samples < time1 + self.time_window
                )
                time_window_idx2 = (self.samples >= time2) & (
                    self.samples < time2 + self.time_window
                )

                # Data for this time bin
                X_train = np.mean(X_train_all[..., time_window_idx1], 2)
                X_test = np.mean(X_test_all[..., time_window_idx2], 2)

                yield X_train, X_test

    def bin_and_custom_split(self, xdata_train, xdata_test, ydata_train, ydata_test, test_size=0.1):
        """
        Takes in train and test data and yields portion of each for purposes of cross-validation.
        Useful if you want data to always be in train, and other data to always be in test.
        e.g. train on color and test on orientation data.

        Keyword arguments:
        xdata_train -- all EEG data for training
        xdata_test -- all EEG data for testing
        ydata_train -- trial labels for training data
        ydata_test -- trial labels for test data
        """
        self.ifold = 0

        for self.ifold in range(self.n_splits):
            xdata_train_binned, ydata_train_binned = self.bin_trials(xdata_train, ydata_train)
            xdata_test_binned, ydata_test_binned = self.bin_trials(xdata_test, ydata_test)

            X_train_all, _, y_train, _ = train_test_split(
                xdata_train_binned, ydata_train_binned, stratify=ydata_train_binned
            )
            _, X_test_all, _, y_test = train_test_split(
                xdata_test_binned,
                ydata_test_binned,
                stratify=ydata_test_binned,
                test_size=test_size,
            )

            yield X_train_all, X_test_all, y_train, y_test


class Classification:
    def __init__(self, wrangl, nsub, num_labels=None, classifier=None):
        """
        Classification and storing of classification outputs.

        Keyword arguments:
        wrangl -- Wrangler object that was used with data
        nsub -- number of subjects in decoding
        num_labels -- number of unique trial labels (default None)
        classifier -- classifier object for decoding (default None). If none, defaults to sklearn's Logistic Regression.
        """
        self.wrangl = wrangl
        self.n_splits = wrangl.n_splits
        self.t = wrangl.t
        if wrangl.num_labels:
            self.num_labels = wrangl.num_labels
        if num_labels:
            self.num_labels = num_labels
        if self.num_labels is None:
            raise Exception("Must provide number of num_labels to Classification")

        self.nsub = nsub

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = LogisticRegression()
        self.scaler = StandardScaler()

        self.acc = np.zeros((self.nsub, np.size(self.t), self.n_splits)) * np.nan
        self.acc_shuff = np.zeros((self.nsub, np.size(self.t), self.n_splits)) * np.nan
        self.conf_mat = (
            np.zeros((self.nsub, np.size(self.t), self.n_splits, self.num_labels, self.num_labels))
            * np.nan
        )
        self.confidence_scores = (
            np.empty((self.nsub, len(self.t), self.n_splits, self.num_labels)) * np.nan
        )

    def standardize(self, X_train, X_test):
        """
        z-score each electrode across trials at this time point. returns standardized train and test data.
        Note: this fits and transforms train data, then transforms test data with mean and std of train data!!!

        Keyword arguments:
        X_train -- time slice of EEG data for training
        X_test -- time slice of EEG data for testing
        """

        # Fit scaler to X_train and transform X_train
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test

    def decode(self, X_train, X_test, y_train, y_test, y_test_shuffle, isub):
        """
        does actual training and testing of classifier after standardizing the data. Also does shuffled testing, confusion matrix, and confidence scores.

        Keyword arguments:
        X_train -- time slice of EEG data for training
        X_test -- time slice of EEG data for testing
        y_train -- trial labels for training data
        y_test -- trial labels for test data
        y_test_shuffle -- shuffled trial labels for shuffle test check
        isub -- index of subject being trained/tested
        """
        ifold = self.wrangl.ifold
        itime = self.wrangl.itime

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, itime, ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, itime, ifold] = self.classifier.score(X_test, y_test_shuffle)

        self.conf_mat[isub, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test)
        )

        confidence_scores = self.classifier.decision_function(X_test)
        for i, ss in enumerate(set(y_test)):
            self.confidence_scores[isub, itime, ifold, i] = np.mean(confidence_scores[y_test == ss])

    def decode_pairwise(self, X_train, X_test, y_train, y_test, y_test_shuffle, isub):
        """
        Same functionality as decode. But results matrices are different shape.
        Used when using group_dict_list and rolling over multiple sets of classes (e.g. 1vs2 and 2vs4)

        Keyword arguments:
        X_train -- time slice of EEG data for training
        X_test -- time slice of EEG data for testing
        y_train -- trial labels for training data
        y_test -- trial labels for test data
        y_test_shuffle -- shuffled trial labels for shuffle test check
        isub -- index of subject being trained/tested
        """

        ifold = self.wrangl.ifold
        itime = self.wrangl.itime
        iss = self.wrangl.iss

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, iss, itime, ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, iss, itime, ifold] = self.classifier.score(X_test, y_test_shuffle)
        self.conf_mat[isub, iss, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test)
        )

    def decode_temp_gen(self, X_train, X_test, y_train, y_test, isub):
        """
        Same functionality as decode. But results matrices are different shape.

        Keyword arguments:
        X_train -- time slice of EEG data for training
        X_test -- time slice of EEG data for testing
        y_train -- trial labels for training data
        y_test -- trial labels for test data
        isub -- index of subject being trained/tested
        """
        ifold = self.wrangl.ifold
        itime1 = self.wrangl.itime1
        itime2 = self.wrangl.itime2

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, itime1, itime2, ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, itime1, itime2, ifold] = self.classifier.score(
            X_test, np.random.permutation(y_test)
        )
        self.conf_mat[isub, itime1, itime2, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test)
        )

    def decode_electrode_subset(self, X_train, X_test, y_train, y_test, isub):
        """
        Same functionality as decode. But results matrices are different shape.

        Keyword arguments:
        X_train -- time slice of EEG data for training
        X_test -- time slice of EEG data for testing
        y_train -- trial labels for training data
        y_test -- trial labels for test data
        isub -- index of subject being trained/tested
        """

        ifold = self.wrangl.ifold
        itime = self.wrangl.itime
        ielec = self.wrangl.ielec

        X_train, X_test = self.standardize(X_train, X_test)

        self.classifier.fit(X_train, y_train)

        self.acc[isub, ielec, itime, ifold] = self.classifier.score(X_test, y_test)
        self.acc_shuff[isub, ielec, itime, ifold] = self.classifier.score(
            X_test, np.random.permutation(y_test)
        )
        self.conf_mat[isub, ielec, itime, ifold] = confusion_matrix(
            y_test, y_pred=self.classifier.predict(X_test)
        )


class Interpreter:
    def __init__(self, clfr=None, subtitle="", output_dir=None, experiment_name=""):
        """
        Visualization and statistical testing.

        Keyword arguments:
        clfr -- Classification object to be interpreted
        subtitle -- subtitle for saving classification results
        output_dir -- directory to output saved results
        experiment_name -- name of experiment
        """
        if clfr is not None:
            self.clfr = clfr
            self.t = clfr.wrangl.t
            self.time_window = clfr.wrangl.time_window
            self.time_step = clfr.wrangl.time_step
            self.trial_bin_size = clfr.wrangl.trial_bin_size
            self.n_splits = clfr.wrangl.n_splits
            self.labels = list(clfr.wrangl.labels)
            self.electrodes = clfr.wrangl.electrodes
            self.acc = clfr.acc
            self.acc_shuff = clfr.acc_shuff
            self.conf_mat = clfr.conf_mat
            self.confidence_scores = clfr.confidence_scores

        import matplotlib

        matplotlib.rcParams["font.sans-serif"] = "Arial"
        matplotlib.rcParams["font.family"] = "sans-serif"
        self.colors = ["royalblue", "firebrick", "forestgreen", "orange", "purple"]

        self.timestr = time.strftime("%Y%m%d_%H%M")
        self.subtitle = subtitle
        self.experiment_name = experiment_name

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = Path("./output")
        self.fig_dir = self.output_dir / "figures"

    def save_results(self, filename=None, additional_values=None, timestamp=True):
        """
        Saves results of classification.

        Keyword arguments:
        filename -- name of file to store results
        additional_values -- additional variables to save
        """
        values = [
            "t",
            "time_window",
            "time_step",
            "trial_bin_size",
            "n_splits",
            "labels",
            "electrodes",
            "acc",
            "acc_shuff",
            "conf_mat",
            "confidence_scores",
        ]
        if additional_values:
            for val in additional_values:
                values.append(val)

        results_dict = {}
        for value in values:
            results_dict[value] = self.__dict__[value]

        if filename is None:
            if timestamp is True:
                filename = self.subtitle + "_" + self.timestr + ".pickle"
            else:
                filename = self.subtitle + ".pickle"
        else:
            filename = filename + ".pickle"

        file_to_save = self.output_dir / filename

        with open(file_to_save, "wb") as fp:
            pickle.dump(results_dict, fp)

    def _load_results(self, filename=None):
        """
        Returns results of classification.

        Keyword arguments:
        filename -- name of file to be loaded
        """

        if filename is None:
            list_of_files = sorted(self.output_dir.glob("*.pickle"))
            file_to_open = max(list_of_files, key=os.path.getctime)
            print("No filename provided. Loading most recent results.")
        else:
            filename = filename + ".pickle"
            file_to_open = self.output_dir / filename

        with open(file_to_open, "rb") as fp:
            results = pickle.load(fp)

        return results

    def load_results(self, filename=None):
        """
        Loads results of classification and updates self.

        Keyword arguments:
        filename -- name of file to be loaded
        """
        results = self._load_results(filename=filename)
        self.__dict__.update(results)

    def combine_interps(self, interp_filenames, overwrite_current_interp=True):
        """
        Combines multiple saved interpreter results for "Subset" comparisons

        Keyword arguments:
        interp_filenames (list) -- list of Interpreter results saved in 'Output' folder
        overwrite_current_interp (bool) -- When true, current interpreter is overwritten before
                                           saved results are combined. Useful when starting off
                                           with blank interp.
        """
        for i, f in enumerate(interp_filenames):
            # to overwrite results
            if (i == 0) & (overwrite_current_interp is True):
                self.load_results(f)

            # to add results anad check compatibility
            else:
                results = self._load_results(f)

                self.check_interp_compatibility(results, f)

                acc = self.acc
                acc_shuff = self.acc_shuff
                results_acc = results["acc"]
                results_acc_shuff = results["acc_shuff"]

                if len(acc.shape) < 4:
                    acc = acc[:, np.newaxis]
                    acc_shuff = acc_shuff[:, np.newaxis]
                if len(results_acc.shape) < 4:
                    results_acc = results_acc[:, np.newaxis]
                    results_acc_shuff = results_acc_shuff[:, np.newaxis]

                self.acc = np.concatenate([acc, results_acc], 1)
                self.acc_shuff = np.concatenate([acc_shuff, results_acc_shuff], 1)

    def check_interp_compatibility(self, results, filename):
        """
        Checks that Interpreter results being combined has the same parameters as self.

        Keyword arguments:
        results -- the results being added to self
        filename -- name of results file being added
        """
        check = np.array(["t", "time_window", "time_step", "trial_bin_size", "n_splits"])

        check_list = np.array([np.all(self.__dict__[c] == results[c]) for c in check])
        no_match = check[~check_list]

        if len(no_match) > 0:
            print(f"WARNING: Attributes {no_match} from {filename} did not match Interpreter!")

    def savefig(self, subtitle="", file_format=[".pdf", ".png"], save=True):
        """
        saves figure as pdf and png in figure directory.

        Keyword arguments:
        subtitle -- subtitle to be included in figure (default '')
        file_format -- list of formats to save figure as (deafult pdf and png)
        save -- if you want to save (default True)
        """
        if save:
            for file in file_format:
                filename = self.subtitle + subtitle + file
                output = self.fig_dir / filename
                plt.savefig(output, bbox_inches="tight", dpi=1000, format=file[1:])
                print(f"Saving {output}")

    @staticmethod
    def get_plot_line(a):
        """
        Takes in 2D array of shape [subjects,time points].
        Returns mean, and upper/lower SEM lines.
        """
        mean = np.mean(a, 0)
        sem = sista.sem(a, 0)
        upper, lower = mean + sem, mean - sem
        return mean, upper, lower

    def plot_acc(
        self,
        subtitle="",
        significance_testing=False,
        stim_time=[0, 250],
        savefig=False,
        title=None,
        ylim=[0.18, 0.55],
        chance_text_xy=(0.82, 0.19),
        stim_text_xy=(0.25, 0.98),
    ):
        """
        Plots classification accuracy
        """
        acc = np.mean(self.acc, 2)
        se = sista.sem(acc, 0)
        acc_mean = np.mean(acc, 0)
        upper_bound, lower_bound = acc_mean + se, acc_mean - se
        acc_shuff = np.mean(self.acc_shuff, 2)
        se_shuff = sista.sem(acc_shuff, 0)
        acc_mean_shuff = np.mean(acc_shuff, 0)
        upper_bound_shuff, lower_bound_shuff = acc_mean_shuff + se_shuff, acc_mean_shuff - se_shuff
        chance = 1 / len(self.labels)
        sig_y = chance - 0.05
        stim_lower = ylim[0] + 0.01
        stim_upper = ylim[1]

        # plotting
        fig, ax = plt.subplots()

        ax.fill_between(
            stim_time, [stim_lower, stim_lower], [stim_upper, stim_upper], color="gray", alpha=0.5
        )
        ax.plot(self.t, np.ones((len(self.t))) * chance, "--", color="gray")
        ax.fill_between(self.t, upper_bound_shuff, lower_bound_shuff, alpha=0.5, color="gray")
        ax.plot(self.t, acc_mean_shuff, color="gray")
        ax.fill_between(self.t, upper_bound, lower_bound, alpha=0.5, color="tomato")
        ax.plot(self.t, acc_mean, color="tab:red", linewidth=2)

        # Significance Testing
        if significance_testing:
            p = np.zeros(len(self.t[self.t > 0]))
            # only test on timepoints after stimulus onset
            for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                # one-sided paired ttest
                _, p[i] = sista.ttest_rel(a=acc[:, t], b=acc_shuff[:, t], alternative="greater")
            # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
            _, corrected_p, _, _ = multipletests(p, method="fdr_bh")
            sig05 = corrected_p < 0.05

            plt.scatter(
                self.t[self.t > 0][sig05] - 10,
                np.ones(sum(sig05)) * (sig_y),
                marker="s",
                s=28,
                c="tab:red",
                label="p < .05",
            )
            sig_timepoints = self.t[self.t > 0][sig05]
            print(f"Significant timepoints: {sig_timepoints}")

        # aesthetics
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks(np.arange(0.1, 1.1, 0.1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(min(self.t), max(self.t))
        plt.ylim(ylim)
        plt.legend(loc="lower right", frameon=False, fontsize=11)

        # labelling
        plt.xlabel("Time from stimulus onset (ms)", fontsize=14)
        plt.ylabel("Classification accuracy", fontsize=14)
        ax.text(
            chance_text_xy[0],
            chance_text_xy[1],
            "Shuffle",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            color="grey",
        )
        ax.text(
            stim_text_xy[0],
            stim_text_xy[1],
            "Stim",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            color="white",
        )
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig("acc" + subtitle, save=savefig)
        plt.show()

        delay_period_acc = np.mean(acc_mean[self.t > stim_time[1]])
        delay_period_sd = np.std(acc_mean[self.t > stim_time[1]])

        print(f"Mean delay accuracy: {delay_period_acc}")
        print(f"Mean delay S.D.: {delay_period_sd}")

    def plot_acc_subset(
        self,
        subset_list,
        chance,
        sig_ys=[0.2125, 0.1875, 0.2],
        subtitle="",
        significance_testing=False,
        stim_time=[0, 250],
        savefig=False,
        title=None,
        ylim=[0.18, 0.55],
        chance_text_xy=(0.82, 0.19),
        stim_text_xy=(0.25, 0.98),
    ):
        """
        plots classification accuracies. Useful when doing classifcation on set size (1vs2,2vs3,3vs4) or electrode subsets (Frontal, Central, Parietal)
        """

        # plotting
        ax = plt.subplot(111)
        stim_lower = ylim[0] + 0.01
        stim_upper = ylim[1]
        ax.fill_between(
            stim_time, [stim_lower, stim_lower], [stim_upper, stim_upper], color="gray", alpha=0.5
        )
        ax.plot(self.t, np.ones((len(self.t))) * chance, "--", color="gray")

        for isubset, subset in enumerate(subset_list):
            color = self.colors[isubset]
            acc = self.acc[:, isubset]
            acc_shuff = self.acc_shuff[:, isubset]

            acc = np.mean(acc, 2)
            se = sista.sem(acc, 0)
            acc_mean = np.mean(acc, 0)
            upper_bound, lower_bound = acc_mean + se, acc_mean - se
            acc_shuff = np.mean(acc_shuff, 2)
            se_shuff = sista.sem(acc_shuff, 0)
            acc_mean_shuff = np.mean(acc_shuff, 0)
            upper_bound_shuff, lower_bound_shuff = (
                acc_mean_shuff + se_shuff,
                acc_mean_shuff - se_shuff,
            )

            ax.fill_between(self.t, upper_bound_shuff, lower_bound_shuff, alpha=0.2, color="gray")
            ax.plot(self.t, acc_mean_shuff, color="gray")
            ax.fill_between(self.t, upper_bound, lower_bound, alpha=0.5, color=color)
            ax.plot(self.t, acc_mean, color=color, label=subset)

            # Significance Testing
            if significance_testing:
                p = np.zeros(len(self.t[self.t > 0]))
                # only test on timepoints after stimulus onset
                for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                    # one-sided paired ttest
                    _, p[i] = sista.ttest_rel(a=acc[:, t], b=acc_shuff[:, t], alternative="greater")

                # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
                _, corrected_p, _, _ = multipletests(p, method="fdr_bh")
                sig05 = corrected_p < 0.05

                plt.scatter(
                    self.t[self.t > 0][sig05] - 10,
                    np.ones(sum(sig05)) * (sig_ys[isubset]),
                    marker="s",
                    s=28,
                    c=color,
                )

                sig_timepoints = self.t[self.t > 0][sig05]
                print(f"{subset} significant timepoints: {sig_timepoints}")

            delay_period_acc = np.mean(acc_mean[self.t > stim_time[1]])
            delay_period_sd = np.std(acc_mean[self.t > stim_time[1]])

            print(f"{subset} mean delay accuracy: {delay_period_acc}")
            print(f"{subset} mean delay S.D.: {delay_period_sd}")

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(handles, subset_list, title="Classification", fontsize=12)
        plt.setp(leg.get_title(), fontsize=12)

        # aesthetics
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks(np.arange(0.1, 1.1, 0.1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(min(self.t), max(self.t))
        plt.ylim(ylim)

        # labelling
        plt.xlabel("Time from stimulus onset (ms)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        ax.text(
            chance_text_xy[0],
            chance_text_xy[1],
            "Shuffle",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            color="grey",
        )
        ax.text(
            stim_text_xy[0],
            stim_text_xy[1],
            "Stim",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            color="white",
        )
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig("acc" + subtitle, save=savefig)
        plt.show()

    def plot_acc_compare_subset(
        self,
        subset_list,
        chance,
        subtitle="",
        significance_testing=False,
        stim_time=[0, 250],
        savefig=False,
        title=None,
        ylim=[0.18, 0.55],
        chance_text_xy=(0.82, 0.19),
        stim_text_xy=(0.25, 0.98),
        legend_title="",
    ):
        """
        plots classification accuracies. Useful when doing classifcation on set size (1vs2,2vs3,3vs4) or electrode subsets (Frontal, Central, Parietal)
        """

        # plotting
        ax = plt.subplot(111)
        stim_lower = ylim[0] + 0.01
        stim_upper = ylim[1]
        ax.fill_between(
            stim_time, [stim_lower, stim_lower], [stim_upper, stim_upper], color="gray", alpha=0.5
        )
        ax.plot(self.t, np.ones((len(self.t))) * chance, "--", color="gray")
        sig_y = chance - 0.05

        for isubset, subset in enumerate(subset_list):
            color = self.colors[isubset]
            acc = self.acc[:, isubset]
            acc_shuff = self.acc_shuff[:, isubset]

            acc = np.mean(acc, 2)
            se = sista.sem(acc, 0)
            acc_mean = np.mean(acc, 0)
            upper_bound, lower_bound = acc_mean + se, acc_mean - se
            acc_shuff = np.mean(acc_shuff, 2)
            se_shuff = sista.sem(acc_shuff, 0)
            acc_mean_shuff = np.mean(acc_shuff, 0)
            upper_bound_shuff, lower_bound_shuff = (
                acc_mean_shuff + se_shuff,
                acc_mean_shuff - se_shuff,
            )

            ax.fill_between(self.t, upper_bound_shuff, lower_bound_shuff, alpha=0.2, color="gray")
            ax.plot(self.t, acc_mean_shuff, color="gray")
            ax.fill_between(self.t, upper_bound, lower_bound, alpha=0.5, color=color)
            ax.plot(self.t, acc_mean, color=color, label=subset)

            delay_period_acc = np.mean(acc_mean[self.t > stim_time[1]])
            delay_period_sd = np.std(acc_mean[self.t > stim_time[1]])
            print(f"{subset} mean delay accuracy: {delay_period_acc}")
            print(f"{subset} mean delay S.D.: {delay_period_sd}")

        # Significance Testing
        if significance_testing:
            acc1 = np.mean(self.acc[:, 0], 2)
            acc2 = np.mean(self.acc[:, 1], 2)

            p = np.zeros(len(self.t[self.t > stim_time[0]]))
            # only test on timepoints after stimulus onset
            for i, t in enumerate(np.arange(len(self.t))[self.t > stim_time[0]]):
                # one-sided paired ttest
                _, p[i] = sista.ttest_rel(a=acc1[:, t], b=acc2[:, t], alternative="greater")

            # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
            _, corrected_p, _, _ = multipletests(p, method="fdr_bh")
            sig05 = corrected_p < 0.05

            plt.scatter(
                self.t[self.t > 0][sig05] - 10,
                np.ones(sum(sig05)) * (sig_y),
                marker="s",
                s=28,
                c="purple",
            )

        delay_period_acc = np.mean(acc_mean[self.t > stim_time[1]])
        delay_period_sd = np.std(acc_mean[self.t > stim_time[1]])

        print(f"mean delay accuracy: {delay_period_acc}")
        print(f"mean delay S.D.: {delay_period_sd}")

        handles, _ = ax.get_legend_handles_labels()
        leg = ax.legend(handles, subset_list, title=legend_title, fontsize=12)
        plt.setp(leg.get_title(), fontsize=12)

        # aesthetics
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks(np.arange(0.1, 1.1, 0.1))
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlim(min(self.t), max(self.t))
        plt.ylim(ylim)

        # labelling
        plt.xlabel("Time from stimulus onset (ms)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        ax.text(
            chance_text_xy[0],
            chance_text_xy[1],
            "Shuffle",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            color="grey",
        )
        ax.text(
            stim_text_xy[0],
            stim_text_xy[1],
            "Stim",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            color="white",
        )
        if title is not None:
            plt.title(title, fontsize=18)

        self.savefig("acc" + subtitle, save=savefig)
        plt.show()

    def plot_confusion_matrix(
        self,
        subtitle="",
        labels=None,
        earliest_t=200,
        time_idx=None,
        lower=0,
        upper=1,
        chance=None,
        savefig=False,
        subplot=111,
        color_map=plt.cm.RdGy_r,
    ):
        """
        plots the confusion matrix for the classifier

        Input:
        self.conf_mat of shape [subjects,timepoints,folds,setsizeA,setsizeB]
        """
        # Use only relevant time points
        if time_idx is None:
            time_idx = self.t > earliest_t
        cm = np.mean(np.mean(np.mean(self.conf_mat[:, time_idx], 2), 1), 0)

        # Normalize
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Get labels and chance level, if necessary
        if labels is None:
            labels = self.labels

        if chance is None:
            chance = (upper - lower) / cm.shape[0]

        # Generate plot
        ax = plt.subplot(subplot)
        ax = sns.heatmap(
            cm,
            center=chance,
            vmin=lower,
            vmax=upper,
            xticklabels=labels,
            yticklabels=labels,
            # non-arg aesthetics
            annot=True,
            square=True,
            annot_kws={"fontsize": 16},
            linewidths=0.5,
            cmap=color_map,
            ax=ax,
        )

        # Clean up axes
        plt.ylabel("True Label", fontsize=16)
        plt.title("Predicted Label", fontsize=16)
        plt.yticks(rotation=0)
        plt.tick_params(
            axis="both",
            which="major",
            labelsize=15,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=True,
            left=False,
        )

        plt.tight_layout()
        self.savefig("conf_mat" + subtitle, save=savefig)
        plt.show()

    def plot_hyperplane(
        self,
        subtitle="",
        stim_time=[0, 250],
        significance_testing_pair=[],
        sig_y=1,
        savefig=False,
        title=None,
        ylim=[-4, 4],
        legend_title="Trial condition",
        legend_pos="lower right",
        label_text_x=-105,
        label_text_ys=[-3.4, 2.8],
        stim_label_xy=[120, 3.5],
        arrow_ys=[-1.1, 1.2],
        train_labels = None
    ):
        """
        Plots the confidence scores of each label.
        """

        ax = plt.subplot(111)
        stim_lower = ylim[0] + 0.01
        stim_upper = ylim[1]
        ax.fill_between(
            stim_time, [stim_lower, stim_lower], [stim_upper, stim_upper], color="gray", alpha=0.5
        )
        ax.plot(self.t, np.zeros((len(self.t))), "--", color="gray")
        ax.axhline(y=0, color="grey", linestyle="--")
        for i in range(self.confidence_scores.shape[-1]):
            # Get means for each condition
            scores = np.mean(self.confidence_scores, 2)[:, :, i]

            mean, upper, lower = self.get_plot_line(scores)
            ax.plot(self.t, mean, self.colors[i], label=self.labels[i])
            ax.fill_between(self.t, upper, lower, color=self.colors[i], alpha=0.5)

        leg = plt.legend(title=legend_title, loc=legend_pos, fontsize=13)
        plt.setp(leg.get_title(), fontsize=13)

        # Significance Testing
        if len(significance_testing_pair) > 0:
            offset = 0
            color = "orange"

            for score_i, score_ii in significance_testing_pair:
                print(score_i)
                p = np.zeros(len(self.t[self.t > 0]))
                # only test on timepoints after stimulus onset
                for i, t in enumerate(np.arange(len(self.t))[self.t > 0]):
                    # one-sided paired ttest
                    _, p[i] = sista.ttest_rel(
                        a=np.mean(self.confidence_scores, 2)[:, t, score_i],
                        b=np.mean(self.confidence_scores, 2)[:, t, score_ii],
                        alternative="less",
                    )

                # Use Benjamini-Hochberg procedure for multiple comparisons, defaults to FDR of .05
                _, corrected_p, _, _ = multipletests(p, method="fdr_bh")
                sig05 = corrected_p < 0.05

                plt.scatter(
                    self.t[self.t > 0][sig05] - 10,
                    np.ones(sum(sig05)) * (sig_y - offset),
                    marker="s",
                    s=33,
                    c=color,
                    label="p < .05",
                )
                sig_timepoints = self.t[self.t > 0][sig05]
                print(f"Significant timepoints: {sig_timepoints}")
                offset += 0.15
                color = "orange"

        # aesthetics
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), fontsize=14)

        plt.xlim(-250, max(self.t))
        plt.ylim(ylim)

        # labelling
        plt.title(title, fontsize=18)
        plt.xlabel("Time from stimulus onset (ms)", fontsize=14)
        plt.ylabel("Distance from hyperplane (a.u.)", fontsize=14)

        if train_labels is None:
            train_labels = []
            train_labels.append(self.labels[0])
            train_labels.append(self.labels[-1])
            
        plt.text(
            label_text_x,
            label_text_ys[0],
            f"Predicted\n{train_labels[0]}",
            fontsize=12,
            ha="center",
            va="top",
        )
        plt.text(
            label_text_x,
            label_text_ys[1],
            f"Predicted\n{train_labels[1]}",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        plt.arrow(
            label_text_x, arrow_ys[0], 0, -1, head_width=45, head_length=0.25, color="k", width=5
        )
        plt.arrow(
            label_text_x, arrow_ys[1], 0, 1, head_width=45, head_length=0.25, color="k", width=5
        )

        plt.tight_layout()
        self.savefig("hyperplane" + subtitle, save=savefig)
        plt.show()

    def temporal_generalizability(
        self, cmap=plt.cm.viridis, lower_lim=0, upper_lim=1, savefig=False
    ):
        """
        Plot temporal generalizability
        Not used in analyses for paper.
        """

        plt.figure()
        plt.imshow(
            np.mean(np.mean(self.acc, 0), 2),
            interpolation="nearest",
            cmap=cmap,
            clim=(lower_lim, upper_lim),
        )
        plt.title("Accuracy for Training/Testing\non Different Timepoints")
        plt.colorbar()

        tick_marks = np.arange(0, len(self.t), 4)

        plt.xticks(tick_marks, self.t[0::4])
        plt.yticks(tick_marks, self.t[0::4])

        plt.xlabel("Testing Timepoint (ms)")
        plt.ylabel("Training Timepoint (ms)")
        plt.gca().invert_yaxis()

        self.savefig("temp_gen_", save=savefig)


class ERP:
    def __init__(self, exp, subtitle="", fig_dir=None):
        self.exp = exp
        self.info = exp.info
        self.xdata_files = exp.xdata_files
        self.ydata_files = exp.ydata_files

        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.subtitle = subtitle

        if fig_dir:
            self.fig_dir = fig_dir
        else:
            self.fig_dir = Path("output/figures")

    def savefig(self, subtitle="", file_format=".pdf", save=True):
        if save:
            filename = self.subtitle + subtitle + self.timestr + file_format
            output = self.fig_dir / filename
            plt.savefig(output, bbox_inches="tight", dpi=1000, format=file_format[1:])
            print(f"Saving {output}")

    def load_all_eeg(self):
        xdata_all = np.empty((self.exp.nsub), dtype="object")
        ydata_all = np.empty((self.exp.nsub), dtype="object")
        for isub in range(self.exp.nsub):
            xdata_all[isub], ydata_all[isub] = self.exp.load_eeg(isub)
        return xdata_all, ydata_all

    def _select_electrodes(self, xdata, electrode_subset=None, electrode_idx=None):
        if electrode_subset is not None:
            # Create index for electrodes to include in plot
            electrode_labels = [
                el
                for n, el in enumerate(self.info["chan_labels"])
                if el.startswith(electrode_subset)
            ]
            electrode_idx = np.in1d(self.info["chan_labels"], electrode_labels)
            xdata = xdata[electrode_idx]
        elif electrode_idx is not None:
            xdata = xdata[electrode_idx]

        return xdata

    def plot_ss(
        self,
        xdata_all,
        ydata_all,
        subtitle="",
        ax=None,
        electrode_subset=None,
        electrode_idx=None,
        condition_subset=None,
        condition_labels=None,
        savefig=False,
        file_format=[".png", ".pdf"],
    ):
        if condition_subset is None:
            condition_subset = np.unique(ydata_all[0])
        ss_data = np.zeros((self.exp.nsub, len(condition_subset), len(self.info["times"])))
        for isub in range(self.exp.nsub):
            xdata = xdata_all[isub]
            ydata = ydata_all[isub]
            for iss, ss in enumerate(condition_subset):
                ss_idx = ydata == ss
                data = np.mean(xdata[ss_idx], 0)
                ss_data[isub, iss] = np.mean(
                    self._select_electrodes(data, electrode_subset, electrode_idx), 0
                )

        if ax is None:
            ax = plt.subplot(111)

        for iss, ss in enumerate(condition_subset):
            if condition_labels is not None:
                cond = condition_labels[iss]
            x = np.mean(ss_data[:, iss], 0)
            se = np.std(ss_data[:, iss], 0) / np.sqrt(self.exp.nsub)

            # ERP
            ax.plot(self.info["times"], x, linewidth=2.5, alpha=0.8, label=cond)
            # SE
            ax.fill_between(self.info["times"], x - se, x + se, alpha=0.3)

        # Grey stim bar
        ax.fill_between([0, 250], [-4, -4], [6, 6], color="gray", alpha=0.5, zorder=0)

        # Hide the right and top spines]
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        # plt.setp(ax.get_xticklabels(), fontsize=14)
        # plt.setp(ax.get_yticklabels(), fontsize=14)

        # Cleaning up plot
        ax.invert_yaxis()
        legend = plt.legend(title="Condition", loc="lower right", fontsize=11)
        plt.setp(legend.get_title(), fontsize=11)
        ax.set_xlabel("Time from Array Onset (ms)")
        ax.set_ylabel("Amplitude (microvolts)")

        if ax is None:
            self.savefig(subtitle=subtitle, save=savefig, file_format=file_format)
            plt.show()
