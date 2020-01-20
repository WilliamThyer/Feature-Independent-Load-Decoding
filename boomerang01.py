"""A basic change detection experiment.

Author - William Thyer thyer@uchicago.edu

https://github.com/WilliamThyer/DualFeatureChangeDetectionEEGandRealtimeEyetracking

Adapted experiment code, originally from Colin Quirk's https://github.com/colinquirk/PsychopyChangeDetection 

If this file is run directly the defaults at the top of the page will be
used. To make simple changes, you can adjust any of these files. For more in depth changes you
will need to overwrite the methods yourself.

Note: this code relies on Colin Quirk's templateexperiments module. You can get it from
https://github.com/colinquirk/templateexperiments and either put it in the same folder as this
code or give the path to psychopy in the preferences.

Classes:
Boomerang01 -- The class that runs the experiment.
"""

import os
import sys
import errno

import json
import random

import numpy as np

import psychopy.core
import psychopy.event
import psychopy.visual
import psychopy.parallel
import psychopy.tools.monitorunittools

import template 
import eyelinker

# Things you probably want to change
number_of_trials_per_block = 100
number_of_blocks = 10
percent_same = 0.5  # between 0 and 1
set_sizes = [1,3]
stim_size = 1.3  # visual degrees, used for X and Y

single_probe = True  # False to display all stimuli at test

keys = ['s', 'd']  # first is same
distance_to_monitor = 90

instruct_text = [(
    'In this experiment you will be remembering colors and orientations.\n\n'
    'In each block, the target feature will switch.\n'
    'Each trial will start with a fixation cross. '
    'Do your best to keep your eyes on it at all times.\n'
    'An array of colored circles with oriented bars will appear.\n'
    'Remember the the target feature and their locations as best you can.\n'
    'Ignore the non-target gray items. You will not be tested on these.\n'
    'After a short delay, another target item will reappear.\n'
    'If it has the SAME target feature as the item in its location before, press the "S" key.\n'
    'If it has a DIFFERENT target feature, press the "D" key.\n'
    'If you are not sure, just take your best guess.\n\n'
    'You will get breaks in between blocks.\n'
    "We'll start with some practice trials.\n\n"
    'Press the "S" key to start.'
)]

data_directory = os.path.join(
    '.', 'Data')

# Things you probably don't need to change, but can if you want to
exp_name = 'B01'

iti_time = 1
sample_time = 0.25
delay_time = 1

allowed_deg_from_fix = 4

# minimum euclidean distance between centers of stimuli in visual angle
# min_distance should be greater than stim_size
min_distance = 4
max_per_quad = 1  # int or None for totally random displays

colors = [
    [1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [1,  1, -1],
]

orients = [0,45,90,135]
stim_idx = [0,1,2,3]

data_fields = [
    'Subject',
    'Block',
    'Trial',
    'Timestamp',
    'Block Feature',
    'Change',
    'SetSize',
    'RT',
    'CRESP',
    'RESP',
    'ACC',
    'LocationTested',
    'Locations',
    'SampleColors',
    'TestColors',
    'SampleOrients',
    'TestOrients'
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
    'Age': 0,
    'Gender': gender_options,
    'Hispanic:': hispanic_options,
    'Race': race_options,
}


# This is the logic that runs the experiment
# Change anything below this comment at your own risk
class Boomerang01(template.BaseExperiment):
    """The class that runs the  experiment.

    Parameters:
    allowed_deg_from_fix -- The maximum distance in visual degrees the stimuli can appear from
        fixation
    colors -- The list of colors (list of 3 values, -1 to 1) to be used in the experiment.
    orients -- The list of orientsations to be used in the experiment.
    stim_idx -- List of indices for colors and orientations.
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
                 number_of_blocks=number_of_blocks, percent_same=percent_same,
                 set_sizes=set_sizes, stim_size=stim_size, colors=colors,
                 orients=orients, stim_idx = stim_idx, keys=keys, 
                 allowed_deg_from_fix=allowed_deg_from_fix,
                 min_distance=min_distance, max_per_quad=max_per_quad,
                 instruct_text=instruct_text, single_probe=single_probe,
                 iti_time=iti_time, sample_time=sample_time,
                 delay_time=delay_time, data_directory=data_directory,
                 questionaire_dict=questionaire_dict, **kwargs):

        self.number_of_trials_per_block = number_of_trials_per_block
        self.number_of_blocks = number_of_blocks
        self.percent_same = percent_same
        self.set_sizes = set_sizes
        self.stim_size = stim_size

        self.colors = colors
        self.orients = orients
        self.stim_idx = stim_idx

        self.iti_time = iti_time
        self.sample_time = sample_time
        self.delay_time = delay_time

        self.keys = keys

        self.allowed_deg_from_fix = allowed_deg_from_fix

        self.min_distance = min_distance

        if max_per_quad is not None and max(self.set_sizes)/4 > max_per_quad:
            raise ValueError('Max per quad is too small.')

        self.max_per_quad = max_per_quad

        self.data_directory = data_directory
        self.instruct_text = instruct_text
        self.questionaire_dict = questionaire_dict

        self.single_probe = single_probe

        self.same_trials_per_set_size = int((
            number_of_trials_per_block / len(set_sizes)) * percent_same)

        if self.same_trials_per_set_size % 1 != 0:
            raise ValueError('Each trial type needs a whole number of trials.')
        else:
            self.diff_trials_per_set_size = (
                number_of_trials_per_block - self.same_trials_per_set_size)
        
        self.color_block = []
        self.orient_block = []
        self.rejection_tracker = np.zeros(5)

        super().__init__(**kwargs)

    def init_tracker(self):
        self.tracker = eyelinker.EyeLinker(
            self.experiment_window,
            self.experiment_name + self.experiment_info['Subject Number'] + '.edf',
            'BOTH')

        self.tracker.initialize_graphics()
        self.tracker.open_edf()
        self.tracker.initialize_tracker()
        self.tracker.send_tracking_settings()
    
    def show_eyetracking_instructions(self):
        self.tracker.display_eyetracking_instructions()
        self.tracker.setup_tracker()

    def start_eyetracking(self, block_num, trial_num):
        """Send block and trial status and start eyetracking recording

        Parameters:
        block_num-- Which block participant is in
        trial_num-- Which trial in block participant is in
        """
        status = 'Block {}, Trial {}'.format(block_num, trial_num)
        self.tracker.send_status(status)

        self.tracker.start_recording()

    def stop_eyetracking(self):
        """Stop eyetracking recording
        """
        self.tracker.stop_recording()

    def realtime_eyetracking(self,wait_time,trial,sampling_rate=.01):
        """Collect real time eyetracking data over a period of time

        Returns eyetracking data

        Parameters:
        wait_time-- How long in ms to collect data for
        sampling_rate-- How many ms between each sample
        """
        start_time = psychopy.core.getTime()
        while psychopy.core.getTime() < start_time + wait_time:

            realtime_data = self.tracker.gaze_data

            reject,eyes = self.check_realtime_eyetracking(realtime_data)
#            reject=False
            if reject:
                if trial['block_feature'] == 0:
                    self.color_block.append(trial)
                else:
                    self.orient_block.append(trial)
                
                self.stop_eyetracking()
                self.display_eyemovement_feedback(eyes)
                return reject
            psychopy.core.wait(sampling_rate)
        
    def check_realtime_eyetracking(self,realtime_data):
        left_eye,right_eye = realtime_data
        lx,ly = left_eye
        rx,ry = right_eye        
        
        eyex = np.nanmean([lx,rx])
        eyey = np.nanmean([ly,ly])
        
        winx,winy = self.experiment_window.size/2
        
        eyex -= winx
        eyey -= winy
        eyes = np.array([eyex,eyey])

        limit_radius = psychopy.tools.monitorunittools.deg2pix(1.5,self.experiment_monitor)###check size of window

        euclid_distance = np.linalg.norm(eyes-np.array([0,0])) 

        if euclid_distance > limit_radius:
            return True,(eyex,eyey)
        else:
            return False,None
            
    def display_eyemovement_feedback(self,eyes):

        psychopy.visual.TextStim(win=self.experiment_window,text='Eye Movement Detected',pos = [0,1], color = [1,-1,-1]).draw()
        psychopy.visual.TextStim(self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        psychopy.visual.Circle(win=self.experiment_window,radius=5,pos=eyes,fillColor='red',units='pix').draw()
        
        self.experiment_window.flip()
        
        psychopy.core.wait(1.5)

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
        self.port = psychopy.parallel.ParallelPort(address=53328)

        
    def send_synced_event(self, code, keyword = "SYNC"):
        """Send port code to EEG and eyetracking message for later synchronization

        Parameters:
        code-- Digits to send
        keyword-- Accompanying sync keyword (matters for later EEGLAB preprocessing)
        """

        message = keyword + ' ' + str(code)

        self.tracker.send_message(message)
        self.port.setData(code)

    def chdir(self):
        """Changes the directory to where the data will be saved.
        """

        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)
    
    def make_block(self,block_num,number_of_trials_per_block = None):
        """Makes a block of trials.

        Returns a shuffled list of trials created by self.make_trial
        """
        if not number_of_trials_per_block:
            number_of_trials_per_block = self.number_of_trials_per_block

        if block_num%((int(self.experiment_info['Subject Number'])%2)+2) == 0:
            block_feature = 0
        else:
            block_feature = 1

        set_size_block = np.tile([1,3], number_of_trials_per_block//2)
        change_block = np.tile([0,0,1,1], number_of_trials_per_block//4)

        block = []
        for set_size,change in zip(set_size_block,change_block):
            block.append(self.make_trial(block_feature,change,set_size))

        block = np.random.permutation(block)

        return block
        
    def make_trial(self, block_feature, change, set_size):
        """Makes a single trial.

        Returns a dictionary of attributes about trial.
        """

        stim_color_idx = np.random.choice(self.stim_idx, size = set_size, replace = False)
        stim_orient_idx = np.random.choice(self.stim_idx, size = set_size, replace = False)
        stim_colors = [self.colors[c] for c in stim_color_idx]
        stim_orients = [self.orients[o] for o in stim_orient_idx]

        locs = self.generate_locations()

        test_location = locs[0]

        if block_feature == 0: #attend color
            if change == 0:
                test_color = self.colors[stim_color_idx[0]]
                cresp = 's'
            else:
                test_color = self.colors[np.random.choice(np.setdiff1d(self.stim_idx,stim_color_idx[0]))]
                cresp = 'd'
            test_orient = self.orients[stim_orient_idx[0]]
        if block_feature == 1: # attend orientation
            if change == 0:
                test_orient = self.orients[stim_orient_idx[0]]
                cresp = 's'
            else:
                test_orient = self.orients[np.random.choice(np.setdiff1d(self.stim_idx,stim_orient_idx[0]))]
                cresp = 'd'
            test_color = self.colors[stim_color_idx[0]]
        
        code = int(str(block_feature+1) + str(change) + str(set_size))

        trial = {
            'code': code,
            'block_feature': block_feature,
            'change': change,
            'set_size': set_size,
            'cresp': cresp,
            'locations': locs,
            'stim_colors': stim_colors,
            'test_color': test_color,
            'stim_orients': stim_orients,
            'test_orient': test_orient,
            'test_location': test_location,
        }

        return trial

    @staticmethod
    def _which_quad(loc):
        """Checks which quad a location is in.

        This method is used by generate_locations to ensure the max_per_quad condition is followed.

        Parameters:
        loc -- A list of two values (x,y) in visual angle.
        """
        if loc[0] < 0 and loc[1] < 0:
            return 0
        elif loc[0] >= 0 and loc[1] < 0:
            return 1
        elif loc[0] < 0 and loc[1] >= 0:
            return 2
        else:
            return 3

    def _too_close(self, attempt, locs):
        """Checks that an attempted location is valid.

        This method is used by generate_locations to ensure the min_distance condition is followed.

        Parameters:
        attempt -- A list of two values (x,y) in visual angle.
        locs -- A list of previous successful attempts to be checked.
        """
        if np.linalg.norm(np.array(attempt)) < self.min_distance:
            return True  # Too close to center

        for loc in locs:
            if np.linalg.norm(np.array(attempt) - np.array(loc)) < self.min_distance:
                return True  # Too close to another square

        return False

    def generate_locations(self):
        """Creates the locations for a trial. A helper function for self.make_trial.

        Returns a list of acceptable locations.

        Parameters:
        set_size -- The number of stimuli for this trial.
        """
        if self.max_per_quad is not None:
            # quad boundries (x1, x2, y1, y2)
            quad_count = [0, 0, 0, 0]

        locs = []
        counter = 0
        while len(locs) < 4:
            counter += 1
            if counter > 1000:
                raise ValueError('Timeout -- Cannot generate locations with given values.')

            attempt = [random.uniform(-self.allowed_deg_from_fix, self.allowed_deg_from_fix)
                       for _ in range(2)]

            if self._too_close(attempt, locs):
                continue

            if self.max_per_quad is not None:
                quad = self._which_quad(attempt)

                if quad_count[quad] < self.max_per_quad:
                    quad_count[quad] += 1
                    locs.append(attempt)
            else:
                locs.append(attempt)

        return locs

    def display_start_block_screen(self,block_feature):
        if block_feature == 0:
            feat = 'color'
        else:
            feat = 'orientation'

        self.display_text_screen(text=f'In this block, attend {feat}\n\n\n\nPress s to continue',key_list=['s'])

    def display_fixation(self,wait_time = None, text = None, key_list = None, realtime_eyetracking = False, trial = None):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        text -- Str that displays above fixation cross. 
        key_list -- If key_list is given, will wait until key press
        trial -- Trial object needed for realtime eyetracking functionality.
        real_time_eyetracking -- Bool for if you want to do realtime eyetracking or not
        """
        
        if text:
            psychopy.visual.TextStim(win=self.experiment_window,text=text,pos = [0,1], color = [1,-1,-1]).draw()

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        self.experiment_window.flip()

        if realtime_eyetracking:
            reject = self.realtime_eyetracking(wait_time=wait_time,trial=trial)
            return reject    
        else:
            if key_list:
                resp = psychopy.event.waitKeys(maxWait=wait_time,keyList=key_list)
                if resp == ['p']:
                    self.display_text_screen(text='Paused',key_list = ['s'])
                    self.display_fixation(wait_time=1)
                elif resp == ['o']:
                    self.tracker.calibrate()
                    self.display_fixation(wait_time=1)
                elif resp == ['escape']:
                    self.quit_experiment()
            else:
                psychopy.core.wait(wait_time)
                
    def draw_stim(self,color,orient,pos):
        circle = psychopy.visual.Circle(
            self.experiment_window, fillColor=color, pos = pos,
            fillColorSpace='rgb', radius=1.3, units='deg', lineColor=None
        )

        orient = psychopy.visual.Rect(
            self.experiment_window, ori = orient, pos = pos,
            fillColor = self.bg_color, fillColorSpace='rgb',
            width=.5, height=2.6, units='deg', lineColor = None
        )
        
        circle.draw()
        orient.draw()
    
    def draw_trak(self,x=930, y=510):
        trak = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor = [1,1,1], 
            fillColorSpace='rgb', radius=20, pos = [x,y], units='pix'
        )
        
        trak.draw()

    def display_stimuli(self, trial, locations, colors, orients, realtime_eyetracking=False):
        """Displays the stimuli. A helper function for self.run_trial.

        Parameters:
        locations -- A list of locations (list of x and y value) describing where the stimuli
            should be displayed.
        colors -- A list of colors describing what should be drawn at each coordinate.
        """

        distractor = psychopy.visual.Circle(
            self.experiment_window, 
            fillColor=[0.17, 0.18, 0.17],
            fillColorSpace='rgb',
            radius=1.13,
            units='deg',
            lineColor=None
        )

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()

        for i in range(len(colors)):
            self.draw_stim(colors[i],orients[i],locations[i])

        for i in range(4-len(colors)):
            distractor.pos = locations[3-i]
            distractor.draw()

        self.draw_trak()

        self.send_synced_event(trial['code'])
        self.experiment_window.flip()
        
        if realtime_eyetracking:
            reject = self.realtime_eyetracking(wait_time=self.sample_time,trial=trial)
            return reject
        else:
            psychopy.core.wait(self.sample_time)

    def display_test(self, change, locations, colors, orients, test_color, test_orient):
        """Displays the test array. A helper function for self.run_trial.

        Parameters:
        change -- Whether the trial is same or different.
        locations -- A list of locations where stimuli should be drawn.
        colors -- The colors that should be drawn at each coordinate.
        test_loc -- The index of the tested stimuli.
        test_color -- The color of the tested stimuli.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
      
        self.draw_stim(color=test_color,orient=test_orient,pos=locations[0])

        self.send_synced_event(3) #success code means no eyetracking rejection
        self.experiment_window.flip()

    def get_response(self):
        """Waits for a response from the participant. A helper function for self.run_trial.

        Returns the pressed key and the reaction time.
        """

        rt_timer = psychopy.core.MonotonicClock()

        keys = self.keys + ['p']

        resp = psychopy.event.waitKeys(keyList=keys, timeStamped=rt_timer)

        return resp[0][0], resp[0][1]*1000  # key and rt in milliseconds

    def send_data(self, data):
        """Updates the experiment data with the information from the last trial.

        This function is seperated from run_trial to allow additional information to be added
        afterwards.

        Parameters:
        data -- A dict where keys exist in data_fields and values are to be saved.
        """
        self.update_experiment_data([data])
    
    def handle_rejection(self,reject):
        self.rejection_tracker = np.roll(self.rejection_tracker,1)
        
        self.rejection_tracker[0] = reject
        print(self.rejection_tracker)
        if np.sum(self.rejection_tracker) == 5:
            
            self.rejection_tracker = np.zeros(5)
            self.display_text_screen(text='Rejected 5 in row\n\nContinue?',key_list = ['y'],bg_color=[0, 0, 255],text_color=[255,255,255])

    def run_trial(self, trial, block_num, trial_num, realtime_eyetracking=False):
        """Runs a single trial.

        Returns the data from the trial after getting a participant response.

        Parameters:
        trial -- The dictionary of information about a trial.
        block_num -- The number of the block in the experiment.
        trial_num -- The number of the trial within a block.
        """
        self.display_fixation(wait_time=np.random.randint(400,601)/1000,trial=trial,key_list=['p','escape','o'])
         
        self.start_eyetracking(block_num = block_num, trial_num = trial_num)
        
        self.send_synced_event(1)
        reject = self.display_fixation(wait_time=self.iti_time,trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        reject = self.display_stimuli(
            trial=trial, locations=trial['locations'], colors=trial['stim_colors'], 
            orients=trial['stim_orients'],  realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        self.send_synced_event(2)
        reject = self.display_fixation(self.delay_time, trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None
        
        self.display_test(
            trial['change'],trial['locations'], trial['stim_colors'], trial['stim_orients'], 
            trial['test_color'], trial['test_orient'])

        resp, rt = self.get_response()
        self.send_synced_event(4)
        self.stop_eyetracking()
        self.handle_rejection(0)
        
        acc = 1 if resp == trial['cresp'] else 0

        data = {
            'Subject': self.experiment_info['Subject Number'],
            'Block': block_num,
            'Trial': trial_num,
            'Timestamp': psychopy.core.getAbsTime(),
            'Block Feature': trial['block_feature'],
            'Change': trial['change'],
            'SetSize': trial['set_size'],
            'RT': rt,
            'CRESP': trial['cresp'],
            'RESP': resp,
            'ACC': acc,
            'LocationTested': trial['test_location'],
            'Locations': json.dumps(trial['locations']),
            'SampleColors': json.dumps(trial['stim_colors']),
            'TestColors': json.dumps(trial['test_color']),
            'SampleOrients':json.dumps(trial['stim_orients']),
            'TestOrients': json.dumps(trial['test_orient'])
        }

        print(f'{block_num+1}, {trial_num+1}')
        print(f'Acc:{acc}')
        return data

    def run_makeup_block(self,block,block_feature,block_num):
        self.tracker.calibrate()
        if block_feature == 0:
            self.color_block = []
        else:
            self.orient_block = []
        
        block_num +=1
        for trial_num, trial in enumerate(block):
            if trial_num == 1:
                self.display_start_block_screen(block_feature)
            elif trial_num % 5 == 0:
                self.tracker.drift_correct()
                
            data = self.run_trial(trial, block_num, trial_num, realtime_eyetracking=True)
            if data:
                self.send_data(data)

        self.save_data_to_csv()
        self.display_text_screen(
            text = 'Block complete.\n\n\n\nPress s to continue.')

    def run(self):
        """Runs the entire experiment.

        This function takes a number of hooks that allow you to alter behavior of the experiment
        without having to completely rewrite the run function. While large changes will still
        require you to create a subclass, small changes like adding a practice block or
        performance feedback screen can be implimented using these hooks. All hooks take in the
        experiment object as the first argument. See below for other parameters sent to hooks.

        Parameters:
        setup_hook -- takes self, executed once the window is open.
        before_first_trial_hook -- takes self, executed after instructions are displayed.
        pre_block_hook -- takes self, block list, and block num
            Executed immediately before block start.
            Can optionally return an altered block list.
        pre_trial_hook -- takes self, trial dict, block num, and trial num
            Executed immediately before trial start.
            Can optionally return an altered trial dict.
        post_trial_hook -- takes self and the trial data, executed immediately after trial end.
            Can optionally return altered trial data to be stored.
        post_block_hook -- takes self, executed at end of block before break screen (including
            last block).
        end_experiment_hook -- takes self, executed immediately before end experiment screen.
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
        self.open_csv_data_file(data_filename = self.experiment_name + self.experiment_info['Subject Number'])
        self.open_window(screen=0)
        self.display_text_screen('Loading...', wait_for_input=False)

        self.init_tracker()

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction, key_list=['s'])
        
        self.setup_eeg()
        self.show_eyetracking_instructions()

        """
        Practice
        """
        block_num = 0
        prac = self.display_text_screen(text = f'Practice block?', key_list=['y','n'])

        while prac == ['y']: 
            
            block = self.make_block(block_num)
            acc = []
            
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_feature'])
                
                data = self.run_trial(trial,block_num,trial_num)      
                acc.append(data['ACC'])

            block_num += 1
            
            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress s to continue.',
                key_list= ['s']
            )
            prac = self.display_text_screen(text = f'Practice block?', key_list=['y','n'])
            
        """
        Experiment
        """
        for block_num in range(self.number_of_blocks):
            block = self.make_block(block_num)
            acc = []
            
            self.tracker.calibrate()
            self.rejection_tracker = np.zeros(5)
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_feature'])
                elif trial_num % 5 == 0:
                    self.tracker.drift_correct()

                data = self.run_trial(trial, block_num, trial_num, realtime_eyetracking=True)
                if data:
                    self.send_data(data)
                    acc.append(data['ACC'])

            self.save_data_to_csv()

            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress s to continue.')

        """
        Makeup Blocks
        """
        while len(self.color_block) > 25:
            self.rejection_tracker = np.zeros(5)
            self.run_makeup_block(self.color_block,0,block_num)
        
        while len(self.orient_block) > 25:
            self.rejection_tracker = np.zeros(5)
            self.run_makeup_block(self.orient_block,1,block_num)
            
        self.display_text_screen(
            'The experiment is now over, please get your experimenter.',
            bg_color=[0, 0, 255], text_color=[255, 255, 255])
        
        self.quit_experiment()


# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = Boomerang01(
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
