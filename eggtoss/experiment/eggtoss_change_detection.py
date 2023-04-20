"""A basic change detection experiment.

Author - William Thyer thyer@uchicago.edu

Adapted experiment code, originally from Colin Quirk's https://github.com/colinquirk/PsychopyChangeDetection

If this file is run directly the defaults at the top of the page will be
used. To make simple changes, you can adjust any of these files. For more in depth changes you
will need to overwrite the methods yourself.

Note: this code relies on Colin Quirk's templateexperiments module. You can get it from
https://github.com/colinquirk/templateexperiments and either put it in the same folder as this
code or give the path to psychopy in the preferences.

Classes:
Eggtoss -- The class that runs the experiment.
"""
import warnings
import os
import sys
import errno
import time
import json
import numpy as np
import math

import psychopy.core
import psychopy.event
import psychopy.visual
import psychopy.parallel
import psychopy.tools.monitorunittools
import template
import eyelinker

warnings.filterwarnings("ignore")

# Things you probably want to change
number_of_trials_per_block = 4  # 52
number_of_ss0_trials_per_block = 0
number_of_blocks = 4
# do 5 RUNS of 4 blocks

percent_same = 0.5  # between 0 and 1

iti_time = .2  # this plus a 400:600 ms jittered iti
sample_time = .5
delay_time = 1
dots_refresh_rate = .01  # in seconds

# Color Setup


def color_convert(color):
    return [round(((n/127.5)-1), 2) for n in color]


color_array_idx = [0, 1, 2, 3, 4, 5, 6]
color_table = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0]
]

rgb_table = []
for colorx in color_array_idx:
    rgb_table.append(color_convert(color_table[colorx]))
grey = color_convert([166, 166, 166])
coherences = [1, 1, 0, 0]

single_probe = True  # False to display all stimuli at test

distance_to_monitor = 90

instruct_text = [(
    'In this experiment you will be remembering colors and dot motion.\n\n'
    'In each block, the target feature will switch.\n'
    'Each trial will start with a fixation cross. '
    'Do your best to keep your eyes on it at all times.\n'
    'An array of colored moving dots will appear.\n'
    'Remember the the target feature and their locations as best you can.\n'
    'Ignore the non-target gray items. You will not be tested on these.\n'
    'After a short delay, another target item will reappear.\n'
    'If it has the SAME target feature as the item in its location before, press the "S" key.\n'
    'If it has a DIFFERENT target feature, press the "D" key.\n'
    'If you are not sure, just take your best guess.\n\n'
    'Press the "S" key to start.'
)]

data_directory = os.path.join(
    '.', 'Data')

exp_name = 'e'

# Things you probably don't need to change, but can if you want to

distance_from_fix = 4

# minimum euclidean distance between centers of stimuli in visual angle
# min_distance should be greater than stim_size
min_distance = 3
keys = ['s', 'd']

data_fields = [
    'Subject',
    'Block',
    'Trial',
    'Timestamp',
    'BlockFeature',
    'SetSize',
    'Change',
    'RT',
    'CRESP',
    'RESP',
    'ACC',
    'TestLocation',
    'TestColor',
    'TestCoherence',
    'TestOrientation',
    'Locations',
    'Colors',
    'Coherences',
    'Orientations'
]

gender_options = [
    'Male',
    'Female',
    'Other/Choose Not To Respond',
]

hispanic_options = [
    'Yes, Hispanic or Latino/a',
    'No, not Hispanic or Latino/a',
    'Choose Not To Respond',
]

race_options = [
    'American Indian or Alaskan Native',
    'Asian',
    'Pacific Islander',
    'Black or African American',
    'White / Caucasian',
    'More Than One Race',
    'Choose Not To Respond',
]

# Add additional questions here
questionaire_dict = {
    'Run': 0,
    'Age': 0,
    'Gender': gender_options,
    'Hispanic:': hispanic_options,
    'Race': race_options,
}

# This is the logic that runs the experiment
# Change anything below this comment at your own risk


class Eggtoss(template.BaseExperiment):
    """The class that runs the  experiment.

    Parameters:
    allowed_deg_from_fix -- The maximum distance in visual degrees the stimuli can appear from
        fixation
    data_directory -- Where the data should be saved.
    delay_time -- The number of seconds between the stimuli display and test.
    instruct_text -- The text to be displayed to the participant at the beginning of the
        experiment.
    iti_time -- The number of seconds in between a response and the next trial.
    keys -- The keys to be used for making a response. First is used for 'same' and the second is
        used for 'different'
    max_per_quad -- The number of stimuli allowed in each quadrant. If None, displays are
        completely random.
    min_distance -- The minimum distance in visual degrees between stimuli.
    number_of_blocks -- The number of blocks in the experiment.
    number_of_trials_per_block -- The number of trials within each block.
    percent_same -- A float between 0 and 1 (inclusive) describing the likelihood of a trial being
        a "same" trial.
    questionaire_dict -- Questions to be included in the dialog.
    sample_time -- The number of seconds the stimuli are on the screen for.
    set_sizes -- A list of all the set sizes. An equal number of trials will be shown for each set
        size.
    single_probe -- If True, the test display will show only a single probe. If False, all the
        stimuli will be shown.
    stim_size -- The size of the stimuli in visual angle.

    Additional keyword arguments are sent to template.BaseExperiment().

    Methods:
    chdir -- Changes the directory to where the data will be saved.
    display_break -- Displays a screen during the break between blocks.
    display_fixation -- Displays a fixation cross.
    display_stimuli -- Displays the stimuli.
    display_test -- Displays the test array.
    generate_locations -- Helper function that generates locations for make_trial
    get_response -- Waits for a response from the participant.
    make_block -- Creates a block of trials to be run.
    make_trial -- Creates a single trial.
    run_trial -- Runs a single trial.
    run -- Runs the entire experiment.
    """

    def __init__(self, number_of_trials_per_block=number_of_trials_per_block,
                 number_of_ss0_trials_per_block=number_of_ss0_trials_per_block,
                 number_of_blocks=number_of_blocks,
                 color_array_idx=color_array_idx, rgb_table=rgb_table, grey=grey,
                 keys=keys, coherences=coherences,
                 distance_from_fix=distance_from_fix,
                 min_distance=min_distance,
                 instruct_text=instruct_text, single_probe=single_probe,
                 iti_time=iti_time, sample_time=sample_time,
                 dots_refresh_rate=dots_refresh_rate,
                 delay_time=delay_time, data_directory=data_directory,
                 questionaire_dict=questionaire_dict, **kwargs):

        self.number_of_trials_per_block = number_of_trials_per_block
        self.number_of_ss0_trials_per_block = number_of_ss0_trials_per_block
        self.number_of_blocks = number_of_blocks
        self.keys = keys

        self.iti_time = iti_time
        self.sample_time = sample_time
        self.delay_time = delay_time
        self.dots_refresh_rate = dots_refresh_rate

        self.distance_from_fix = distance_from_fix
        self.min_distance = min_distance

        self.data_directory = data_directory
        self.instruct_text = instruct_text
        self.questionaire_dict = questionaire_dict

        self.single_probe = single_probe

        self.rgb_table = rgb_table
        self.grey = grey
        self.color_array_idx = color_array_idx
        self.coherences = coherences

        self.rej_counter = []
        self.rejection_tracker = np.zeros(5)

        super().__init__(**kwargs)

    def init_tracker(self):
        self.tracker = eyelinker.EyeLinker(
            self.experiment_window,
            self.experiment_name + '_' + self.experiment_info['Subject Number'].zfill(
                2) + '_' + str(self.experiment_info['Run']) + '.edf',
            'BOTH')

        self.tracker.initialize_graphics()
        self.tracker.open_edf()
        self.tracker.initialize_tracker()
        self.tracker.send_tracking_settings()

    def init_stim(self, location=(0, 0), coherence=0, dir=90, color=[0, 0, 0]):

        dots = psychopy.visual.DotStim(
            fieldShape='circle', fieldSize=(3, 3),
            fieldPos=location,
            noiseDots='direction', signalDots='same',
            speed=.06, dotLife=4,
            dotSize=7.5, color=color,
            coherence=coherence, dir=dir,
            nDots=100, autoLog=True,
            win=self.experiment_window)

        return dots

    def show_eyetracking_instructions(self):
        self.tracker.display_eyetracking_instructions()
        self.tracker.setup_tracker()

    def start_eyetracking(self, block_num, trial_num):
        """Send block and trial status and start eyetracking recording

        Parameters:
        block_num-- Which block participant is in
        trial_num-- Which trial in block participant is in
        """
        status = f'Block {block_num+1}, Trial {trial_num+1}'
        self.tracker.send_status(status)

        self.tracker.start_recording()

    def stop_eyetracking(self):
        """Stop eyetracking recording
        """
        self.tracker.stop_recording()

    def realtime_eyetracking(self, trial, wait_time, sampling_rate=.01, draw_stim=False, stim=None):
        """Collect real time eyetracking data over a period of time

        Returns eyetracking data

        Parameters:
        wait_time-- How long in ms to collect data for
        sampling_rate-- How many ms between each sample
        """
        start_time = psychopy.core.getTime()
        while psychopy.core.getTime() < start_time + wait_time:

            if draw_stim:
                self.draw_stimuli(stim)

            realtime_data = self.tracker.gaze_data

            reject, eyes = self.check_realtime_eyetracking(realtime_data)
#            reject = False

            if reject:
                self.rej_counter.append(trial)

                print(
                    f'# of rejected trials this block: {len(self.rej_counter)}')

                self.stop_eyetracking()
                self.display_eyemovement_feedback(eyes)

                self.tracker.drift_correct()

                return reject

            psychopy.core.wait(sampling_rate)

    def check_realtime_eyetracking(self, realtime_data):
        left_eye, right_eye = realtime_data
        if left_eye:
            lx, ly = left_eye
        if right_eye:
            rx, ry = right_eye
        if (not left_eye) & (not right_eye):
            return False, None

        eyex = np.nanmean([lx, rx])
        eyey = np.nanmean([ly, ry])

        winx, winy = self.experiment_window.size/2

        eyex -= winx
        eyey -= winy
        eyes = np.array([eyex, eyey])

        limit_radius = psychopy.tools.monitorunittools.deg2pix(
            1.5, self.experiment_monitor)

        euclid_distance = np.linalg.norm(eyes-np.array([0, 0]))

        if euclid_distance > limit_radius:
            return True, (eyex, eyey)
        else:
            return False, None

    def display_eyemovement_feedback(self, eyes):

        psychopy.visual.TextStim(win=self.experiment_window,
                                 text='Eye Movement Detected',
                                 pos=[0, 1], color=[1, -1, -1]).draw()

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()

        psychopy.visual.Circle(win=self.experiment_window, radius=16,
                               pos=eyes, fillColor='red', units='pix').draw()

        self.experiment_window.flip()
        psychopy.core.wait(.5)

    def handle_rejection(self, reject):
        self.rejection_tracker = np.roll(self.rejection_tracker, 1)
        self.rejection_tracker[0] = reject

        if np.sum(self.rejection_tracker) == 5:
            self.rejection_tracker = np.zeros(5)
            self.display_text_screen(text='Rejected 5 in row\n\nContinue?', keyList=[
                                     'y'], bg_color=[0, 0, 255], text_color=[255, 255, 255])

    def kill_tracker(self):
        """Turns off eyetracker and transfers EDF file
        """
        self.tracker.set_offline_mode()
        self.tracker.close_edf()
        self.tracker.transfer_edf()
        self.tracker.close_connection()

    def setup_eeg(self):
        """ Connects the parallel port for EEG port code
        """
        try:
            self.port = psychopy.parallel.ParallelPort(address=53328)
        except:
            self.port = None
            print('No parallel port connected. Port codes will not send!')

    def send_synced_event(self, code, keyword="SYNC"):
        """Send port code to EEG and eyetracking message for later synchronization

        Parameters:
        code-- Digits to send
        keyword-- Accompanying sync keyword (matters for later EEGLAB preprocessing)
        """

        message = keyword + ' ' + str(code)

        if self.port:
            self.port.setData(code)
            psychopy.core.wait(.005)
            self.port.setData(0)
            self.tracker.send_message(message)

    def chdir(self):
        """Changes the directory to where the data will be saved.
        """

        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)

    def make_block(self, block_num, number_of_trials_per_block=None, block_feature=None, num_ss0_trials=0):
        """Makes a block of trials.

        Returns a shuffled list of trials created by self.make_trial
        """
        if not number_of_trials_per_block:
            number_of_trials_per_block = self.number_of_trials_per_block

        if block_feature is None:
            block_feature = (
                int(self.experiment_info['Subject Number'])+block_num) % 2

        set_size_block = np.tile([1, 2], number_of_trials_per_block//2)
        change_block = np.tile([0, 0, 1, 1], number_of_trials_per_block//4)
        if num_ss0_trials > 0:
            set_size_block = np.append(set_size_block, [0]*num_ss0_trials)
            change_block = np.append(change_block, [0, 1]*num_ss0_trials)

        block = []
        for set_size, change in zip(set_size_block, change_block):
            block.append(self.make_trial(
                block_num, block_feature, change, set_size))

        block = np.random.permutation(block)

        return block

    def make_trial(self, block_num, block_feature, change, set_size):
        """Makes a single trial.

        Returns a dictionary of attributes about trial.
        """

        stim_color_idx = np.random.choice(
            self.color_array_idx, size=3, replace=False)
        stim_coherences = np.random.choice(
            self.coherences, size=3, replace=False)
        stim_colors = [self.rgb_table[c] for c in stim_color_idx]
        stim_oris = np.random.choice(np.arange(0, 359), 3, replace=False)

        locs = self.generate_locations()

        test_location = locs[0]
        test_ori = np.random.choice(np.arange(0, 359), 1, replace=False)

        if block_feature == 0:  # attend color
            if change == 0:
                test_color = self.rgb_table[stim_color_idx[0]]
                cresp = 's'
            else:
                test_color = self.rgb_table[np.random.choice(
                    np.setdiff1d(self.color_array_idx, stim_color_idx[0]))]
                cresp = 'd'
            test_coherence = stim_coherences[0]
        if block_feature == 1:  # attend coherence
            if change == 0:
                test_coherence = stim_coherences[0]
                cresp = 's'
            else:
                test_coherence = np.random.choice(
                    np.setdiff1d(stim_coherences, stim_coherences[0]))
                cresp = 'd'
            test_color = self.rgb_table[stim_color_idx[0]]

        if set_size == 0:
            cresp, stim_colors, test_color, stim_coherences, test_coherence, change = self.make_ss0_trial()

        code = int(str(block_feature+1) + str(set_size))

        trial = {
            'block_num': block_num,
            'code': code,
            'block_feature': block_feature,
            'change': change,
            'set_size': set_size,
            'cresp': cresp,
            'locations': locs,
            'stim_colors': stim_colors,
            'stim_coherences': stim_coherences,
            'stim_oris': stim_oris,
            'test_location': test_location,
            'test_color': test_color,
            'test_coherence': test_coherence,
            'test_ori': test_ori,
        }

        return trial

    def make_ss0_trial(self):

        cresp = 's'
        stim_colors = [self.grey]*3
        stim_coherences = [0]*3
        change = 0
        test_color = self.grey
        test_coherence = 0

        return cresp, stim_colors, test_color, stim_coherences, test_coherence, change

    @staticmethod
    def make_coords(lower, upper):
        return np.random.choice(np.arange(lower, upper), size=1)[0]

    def generate_locations(self, num_locs=3):

        quads = [((-3, -1.5), (-3, -1.5)), ((-3, -1.5), (1.5, 3)),
                 ((1.5, 3), (1.5, 3)), ((1.5, 3), (-3, -1.5))]

        locs = []
        for loc in quads:
            x = self.make_coords(loc[0][0], loc[0][1])
            y = self.make_coords(loc[1][0], loc[1][1])
            locs.append((x, y))

        locs_subset = np.random.choice(range(4), size=num_locs, replace=False)
        locs = [locs[i] for i in locs_subset]

        return locs

    def display_start_block_screen(self, block_feature):
        if block_feature == 0:
            feat = 'color'
        else:
            feat = 'motion'

        self.display_text_screen(
            text=f'In this block, attend {feat}\n\n\n\nPress s to continue', keyList=['s'])

    def display_fixation(self, wait_time=None, text=None, keyList=None, realtime_eyetracking=False, trial=None):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        text -- Str that displays above fixation cross.
        keyList -- If keyList is given, will wait until key press
        trial -- Trial object needed for realtime eyetracking functionality.
        real_time_eyetracking -- Bool for if you want to do realtime eyetracking or not
        """

        if text:
            psychopy.visual.TextStim(win=self.experiment_window, text=text, pos=[
                                     0, 1], color=[1, -1, -1]).draw()

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()

        self.experiment_window.flip()

        if realtime_eyetracking:
            reject = self.realtime_eyetracking(trial, wait_time=wait_time)
            return reject
        else:
            if keyList:
                resp = psychopy.event.waitKeys(
                    maxWait=wait_time, keyList=keyList)
                if resp == ['p']:
                    self.display_text_screen(text='Paused', keyList=['s'])
                    self.display_fixation(wait_time=1)
                elif resp == ['o']:
                    self.tracker.calibrate()
                    self.display_fixation(wait_time=1)
                elif resp == ['escape']:
                    resp = self.display_text_screen(
                        text='Are you sure you want to exit?', keyList=['y', 'n'])
                    if resp == ['y']:
                        self.tracker.transfer_edf()
                        self.quit_experiment()
                    else:
                        self.display_fixation(wait_time=1)
            else:
                psychopy.core.wait(wait_time)

    def create_stim(self, trial):

        dots_list = []

        # draw stim
        for istim in range(trial['set_size']):
            dots = self.init_stim(trial['locations'][istim], trial['stim_coherences']
                                  [istim], trial['stim_oris'][istim], trial['stim_colors'][istim])
            dots_list.append(dots)

        # draw distractor
        for istim in range(trial['set_size'], 3):
            dots = self.init_stim(
                trial['locations'][istim], trial['stim_coherences'][istim], trial['stim_oris'][istim], self.grey)
            dots_list.append(dots)

        return dots_list

    def draw_trak(self, x=930, y=510):
        trak = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor=[1, 1, 1],
            fillColorSpace='rgb', radius=20, pos=[x, y], units='pix'
        )

        trak.draw()

    def draw_stimuli(self, dots_list):
        for dots in dots_list:
            dots.draw()
        self.draw_trak()
        self.fix.draw()
        self.experiment_window.flip()

    def display_stimuli(self, trial, realtime_eyetracking=False):
        """Displays the stimuli. A helper function for self.run_trial.

        Parameters:
        locations -- A list of locations (list of x and y value) describing where the stimuli
            should be displayed.
        colors -- A list of colors describing what should be drawn at each coordinate.
        """

        dots_list = self.create_stim(trial)

        self.send_synced_event(trial['code'])

        if realtime_eyetracking is False:
            start = time.time()
            while time.time() - start < self.sample_time:
                self.draw_stimuli(dots_list)
                psychopy.core.wait(self.dots_refresh_rate)
        else:
            reject = self.realtime_eyetracking(
                trial, wait_time=self.sample_time, sampling_rate=self.dots_refresh_rate, draw_stim=True, stim=dots_list)
            return reject

        # self.send_synced_event(trial['code'])
        self.experiment_window.flip()

    def display_test(self, trial):
        """Displays the test array. A helper function for self.run_trial.

        Parameters:
        change -- Whether the trial is same or different.
        locations -- A list of locations where stimuli should be drawn.
        colors -- The colors that should be drawn at each coordinate.
        test_loc -- The index of the tested stimuli.
        test_color -- The color of the tested stimuli.
        """

        # success code means no eyetracking rejection
        self.send_synced_event(3)
        self.fix.draw()

        dots = self.init_stim(
            trial['test_location'], trial['test_coherence'], trial['test_ori'], trial['test_color'])

        rt_timer = psychopy.core.MonotonicClock()
        resp = []
        while (len(resp) == 0) or (rt_timer.getTime()*1000 < 100):
            dots.draw()
            self.draw_trak()
            self.fix.draw()
            self.experiment_window.flip()
            resp = psychopy.event.getKeys(
                keyList=self.keys, timeStamped=rt_timer)
            psychopy.core.wait(self.dots_refresh_rate)

        return resp[0][0], resp[0][1]*1000  # key and rt in milliseconds

    def draw_instructions_examples(self):

        locations = [[0, 4.5],
                     [-8.5, 0], [-4.5, 0], [-8.5, -4.5], [-4.5, -4.5],
                     [4.5, 0], [8.5, 0], [4.5, -4.5], [8.5, -4.5]]
        colors = [self.grey,
                  self.rgb_table[1], self.rgb_table[3], self.rgb_table[4], self.rgb_table[4],
                  self.rgb_table[2], self.rgb_table[2], self.rgb_table[5], self.rgb_table[5]]
        coherences = [0, 0, 0, 1, 1, 1, 1, 0, 1]
        oris = [0, 0, 0, 170, 50, 60, 140, 20, 90]

        locations = [[-8.5, 0],  [-4.5, -4.5], [-2.5,.5]]
        colors = [self.rgb_table[1], self.rgb_table[2],self.rgb_table[3]]
        coherences = [0, 1, 1]
        oris = [0, 90, 15]

        dots_list = []
        for istim in range(3):
            dots = self.init_stim(
                locations[istim], coherences[istim], oris[istim], colors[istim])
            dots_list.append(dots)

        # draw stim
        resp = []
        while (len(resp) == 0):
            for dots in dots_list:
                dots.draw()
            self.experiment_window.flip()
            resp = psychopy.event.getKeys(keyList=self.keys)
            psychopy.core.wait(self.dots_refresh_rate)

    def send_data(self, data):
        """Updates the experiment data with the information from the last trial.

        This function is seperated from run_trial to allow additional information to be added
        afterwards.

        Parameters:
        data -- A dict where keys exist in data_fields and values are to be saved.
        """
        self.update_experiment_data([data])

    def run_trial(self, trial, trial_num, realtime_eyetracking=False):
        """Runs a single trial.

        Returns the data from the trial after getting a participant response.

        Parameters:
        trial -- The dictionary of information about a trial.
        block_num -- The number of the block in the experiment.
        trial_num -- The number of the trial within a block.
        """
        self.display_fixation(wait_time=np.random.randint(
            400, 601)/1000, trial=trial, keyList=['p', 'escape', 'o'])
        self.start_eyetracking(
            block_num=trial['block_num'], trial_num=trial_num)

        self.send_synced_event(1)
        reject = self.display_fixation(
            wait_time=self.iti_time, trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        reject = self.display_stimuli(
            trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        self.send_synced_event(2)
        reject = self.display_fixation(
            self.delay_time, trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        resp, rt = self.display_test(trial)

        self.send_synced_event(4)
        self.stop_eyetracking()
        self.handle_rejection(0)

        acc = 1 if resp == trial['cresp'] else 0

        data = {
            'Subject': self.experiment_info['Subject Number'],
            'Block': trial['block_num'],
            'BlockFeature': trial['block_feature'],
            'Trial': trial_num,

            'Timestamp': psychopy.core.getAbsTime(),
            'Change': trial['change'],
            'SetSize': trial['set_size'],
            'Colors': json.dumps(trial['stim_colors']),
            'Coherences': trial['stim_coherences'],
            'Orientations': trial['stim_oris'],
            'RT': rt,
            'CRESP': trial['cresp'],
            'RESP': resp,
            'ACC': acc,
            'Locations': json.dumps(trial['locations']),
            'TestLocation': json.dumps(trial['test_location']),
            'TestColor': trial['test_color'],
            'TestCoherence': trial['test_coherence'],
            'TestOrientation': trial['test_ori'],
        }

        print(f'{trial["block_num"]+1}, {trial_num+1}')
        print(f'Acc:{acc}')
        return data

    def run_makeup_block(self, block_feature, block_num, trial_num):

        block = []
        for trial in self.rej_counter:

            set_size, change = trial['set_size'], trial['change']
            block.append(self.make_trial(
                block_num, block_feature, change, set_size))

        block = np.random.permutation(block)

        for trial in block:

            trial_num += 1

            if trial_num % 5 == 0:
                self.tracker.drift_correct()

            data = self.run_trial(trial, trial_num, realtime_eyetracking=True)
            if data:
                self.send_data(data)

    def run(self):
        """Runs the entire experiment.

        This function takes a number of hooks that allow you to alter behavior of the experiment
        without having to completely rewrite the run function. While large changes will still
        require you to create a subclass, small changes like adding a practice block or
        performance feedback screen can be implimented using these hooks. All hooks take in the
        experiment object as the first argument. See below for other parameters sent to hooks.

        Parameters:
        """

        """
        Setup and Instructions
        """
        self.chdir()

        ok = self.get_experiment_info_from_dialog(self.questionaire_dict)

        if not ok:
            print('Experiment has been terminated.')
            sys.exit(1)

        self.save_experiment_info()
        data_filename = (
            f'{self.experiment_name}_{self.experiment_info["Subject Number"].zfill(2)}_{str(self.experiment_info["Run"])}')
        self.open_csv_data_file(data_filename)
        self.open_window(screen=0)
        self.display_text_screen('Loading...', wait_for_input=False)

        self.init_tracker()
        self.fix = psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1])

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction, keyList=['s'])
        self.draw_instructions_examples()

        self.show_eyetracking_instructions()

        """
        Practice
        """
        self.port = None
        prac = self.display_text_screen(
            text=f'Motion Prac: Press M\n\nColor Prac: Press C', keyList=['m', 'c', 'n'])

        block_num = 0
        while prac != ['n']:

            if prac == ['m']:
                block_feature = 1
            else:
                block_feature = 0

            block = self.make_block(
                block_num=block_num, block_feature=block_feature, number_of_trials_per_block=12, num_ss0_trials=3)
            acc = []

            for trial_num, trial in enumerate(block):

                if trial_num == 0:
                    self.display_start_block_screen(trial['block_feature'])

                data = self.run_trial(trial, trial_num=trial_num)
                acc.append(data['ACC'])

            self.display_text_screen(
                text=f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress s to continue.', keyList=['s'])

            prac = self.display_text_screen(
                text=f'Motion Prac: Press M\n\nColor Prac: Press C', keyList=['m', 'c', 'n'])

        block_num = 0
        while prac != ['n']:

            if prac == ['m']:
                block_feature = 1
            else:
                block_feature = 0

            block = self.make_block(
                block_num=block_num, block_feature=block_feature, number_of_trials_per_block=12, num_ss0_trials=3)
            acc = []

            for trial_num, trial in enumerate(block):

                if trial_num == 0:
                    self.display_start_block_screen(trial['block_feature'])

                data = self.run_trial(trial, trial_num=trial_num)
                acc.append(data['ACC'])

            self.display_text_screen(
                text=f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress s to continue.', keyList=['s'])

        """
        Experiment
        """
        self.setup_eeg()

        for block_num in range(self.number_of_blocks+1):

            if block_num <= 3:
                block = self.make_block(block_num=block_num)
            else:
                # end of run, do block with set size 0.
                # alternate block feature of this block every run
                block_feature = self.experiment_info['Run'] % 2
                block = self.make_block(
                    block_num=block_num, number_of_trials_per_block=50, num_ss0_trials=25, block_feature=block_feature)
            self.rej_counter = []
            acc = []

            self.tracker.calibrate()

            for trial_num, trial in enumerate(block):

                if trial_num == 0:
                    self.display_start_block_screen(trial['block_feature'])
                if trial_num % 5 == 0:
                    self.tracker.drift_correct()

                data = self.run_trial(
                    trial, trial_num, realtime_eyetracking=True)
                if data:
                    self.send_data(data)
                    if trial['set_size'] != 0:
                        acc.append(data['ACC'])

            if len(self.rej_counter) > 5:
                print('Doing makeup block')
                self.run_makeup_block(
                    trial['block_feature'], block_num, trial_num)

            self.display_text_screen(
                text=f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress s to continue.', keyList=['s'])

        self.save_data_to_csv()

        """
        End of Experiment
        """
        self.display_text_screen('The run is now over.', bg_color=[
                                 0, 0, 255], text_color=[255, 255, 255])

        self.tracker.transfer_edf()
        self.quit_experiment()


# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = Eggtoss(
        # BaseExperiment parameters
        experiment_name=exp_name,
        data_fields=data_fields,
        monitor_distance=distance_to_monitor,
        # Custom parameters go here
    )

    try:
        exp.run()
    except Exception as e:
        exp.kill_tracker()
        raise e
