# DualFeatureChangeDetectionEEGandRealtimeEyetracking
Experiment code for "Boomerang" and "Eggtoss" studies. Uses https://github.com/colinquirk/templateexperiments to build experiments. Includes realtime eyetracking rejection and synchronous EEG port codes.

# Experiments

## Boomerang

Experiment 1. Color and orientation change detection task.

## Eggtoss

Experiment 2. Color and motion coherence change detection task.

# Artifact rejection

## eegreject.m

Main script that handles alignment, epoching, other preprocessing, and artifact rejection.

## align_channels.m

Realign eyetracking, EOG, and stimtrak to make channels more visible during inspection of EEG for manual rejection.

## data_from_checked_eeg.m

Pull code from matlab and save it in .mat file for later analysis in Python.
