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
Discus -- The class that runs the experiment.
"""
import os
import sys
import errno

import json
import random
import copy

import numpy as np
import math

import psychopy.core
import psychopy.event
import psychopy.visual
import psychopy.parallel
import psychopy.tools.monitorunittools
import template 
import eyelinker

import warnings
warnings.filterwarnings("ignore")

# Things you probably want to change
number_of_trials_per_block = None #FIX
percent_same = 0.5  # not used
conditions = ['ss1_vdis','ss1_dis','ss1_sim','ss2']
conditions_dict = {
    'ss1_vdis': {'condition': 'ss1_vdis', 'set_size': 1, 'num_distractors': 1,'code': 11},
    'ss1_dis': {'condition': 'ss1_dis', 'set_size': 1,'num_distractors': 1, 'code': 12},
    'ss1_sim': {'condition': 'ss1_sim', 'set_size': 1,'num_distractors': 1, 'code': 13},
    'ss2': {'condition': 'ss2', 'set_size': 2,'num_distractors': 0, 'code': 20},
}

#Color Setup
def color_convert(color):
    return [round(((n/127.5)-1), 2) for n in color]

color_array_idx = [0,1,2,3,4,5,6]
color_table =[
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
grey = color_convert([163,163,163])

square_target_size = (1,1)  # visual degrees, used for X and Y
similar_distractor_size = (.6666, 1.5)
dissimilar_distractor_size = (.55, 1.818)

single_probe = True  # False to display all stimuli at test

distance_to_monitor = 90

instruct_text = [(
    'In this experiment you will be remembering colors.\n\n'
    'Each trial will start with a fixation cross. '
    'Do your best to keep your eyes on it at all times.\n'
    'An array of colored squares will appear.\n'
    'Remember the colors and their locations as best you can.\n'
    'Ignore the non-target rectangles items. You will not be tested on these.\n'
    'After a short delay, color wheel will appear in the location of a target.\n'
    'Use you mouse to click on the color wheel.\n'
    'Recreate the color of the square that was in that location previously.\n'
    'If you are not sure, just take your best guess.\n\n'
    'You will get breaks in between blocks.\n'
    "We'll start with some practice trials.\n\n"
    'Press the "Space" key to start.'
)]

data_directory = os.path.join(
    '.', 'Data')

exp_name = 'd01'

# Things you probably don't need to change, but can if you want to
iti_time = .2 #this plus a 400:600 ms jittered iti
sample_time = 0.15
delay_time = .85

colorwheel_path = './colors.json'
min_color_dist = 25

distance_from_fix = 4

# minimum euclidean distance between centers of stimuli in visual angle
# min_distance should be greater than stim_size
min_distance = 2
keys = ['s','d']

data_fields = [
    'Subject',
    'Block',
    'Run',
    'Condition',
    'Trial',
    'Timestamp',
    'RT',
    'TestLocation',
    'ColorIndex',
    'TrueColor',
    'RespColor',
    'RespIndex',
    'Error',
    'Locations',
    'Quintants',
    'Colors',
    'NumDistractors'
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
class Discus(template.BaseExperiment):
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
                 color_array_idx = color_array_idx, rgb_table=rgb_table,grey=grey,
                 conditions=conditions, conditions_dict = conditions_dict,
                 square_target_size = square_target_size, similar_distractor_size = similar_distractor_size,
                 dissimilar_distractor_size = dissimilar_distractor_size, keys = keys,
                 distance_from_fix=distance_from_fix,
                 min_distance=min_distance,
                 instruct_text=instruct_text, single_probe=single_probe,
                 iti_time=iti_time, sample_time=sample_time,
                 delay_time=delay_time, data_directory=data_directory,
                 questionaire_dict=questionaire_dict,
                 colorwheel_path=colorwheel_path, min_color_dist = 25,
                 **kwargs):

        self.number_of_trials_per_block = number_of_trials_per_block
        self.conditions = conditions
        self.conditions_dict = conditions_dict
        self.square_target_size = square_target_size
        self.similar_distractor_size = similar_distractor_size 
        self.dissimilar_distractor_size = dissimilar_distractor_size 

        self.keys = keys

        self.iti_time = iti_time
        self.sample_time = sample_time
        self.delay_time = delay_time

        self.distance_from_fix = distance_from_fix 
        self.min_distance = min_distance

        self.data_directory = data_directory
        self.instruct_text = instruct_text
        self.questionaire_dict = questionaire_dict

        self.single_probe = single_probe

        self.rgb_table = rgb_table
        self.grey = grey
        self.color_array_idx = color_array_idx
        
        self.rej_counter = []
        self.rejection_tracker = np.zeros(5)

        self.quints_dict = {0: (54,126), 1: (126,198), 2: (198, 270), 3: (270, 342), 4: (342, 54)}
        
        self.min_color_dist = min_color_dist
        self.color_wheel = self._load_color_wheel(colorwheel_path)
        self.mouse = None

        super().__init__(**kwargs)

    def init_tracker(self):
        self.tracker = eyelinker.EyeLinker(
            self.experiment_window,
            self.experiment_name + '_' + self.experiment_info['Subject Number'].zfill(2) + '_' + str(self.experiment_info['Run']) + '.edf',
            'BOTH')

        self.tracker.initialize_graphics()
        self.tracker.open_edf()
        self.tracker.initialize_tracker()
        self.tracker.send_tracking_settings()
    
    def init_stim(self):

        self.target = psychopy.visual.Rect(
            self.experiment_window, lineColor=None, fillColor=[0,0,0], fillColorSpace='rgb', 
            width=square_target_size[0], height=square_target_size[1],units='deg')
        self.sim_distractor = psychopy.visual.Rect(
            self.experiment_window, lineColor=None, fillColor=[0,0,0], fillColorSpace='rgb', 
            width=similar_distractor_size[0], height=similar_distractor_size[1],units='deg')
        self.dis_distractor = psychopy.visual.Rect(
            self.experiment_window, lineColor=None, fillColor=[0,0,0], fillColorSpace='rgb', 
            width=dissimilar_distractor_size[0], height=dissimilar_distractor_size[1],units='deg')

        mask = np.zeros([100, 1])
        mask[-30:] = 1

        rotated_wheel = np.roll(self.color_wheel, 0, axis=0)
        tex = np.repeat(rotated_wheel[np.newaxis, :, :], 360, 0)

        self.color_wheel_stim = psychopy.visual.RadialStim(
            self.experiment_window, tex=tex, mask=mask, pos=(0,0), angularRes=256,
            angularCycles=1, interpolate=False, size=self.square_target_size[0] * 2)
    
    def convert_color_value(self, color, deconvert=False):
        """Converts a list of 3 values from 0 to 255 to -1 to 1.

        Parameters:
        color -- A list of 3 ints between 0 and 255 to be converted.
        """

        if deconvert is True:
            return [round((((n - -1) * 255) / 2) + 0,1) for n in color]
        else:
            return [round((((n - 0) * 2) / 255) + -1,3) for n in color]
    
    def _load_color_wheel(self,path):

        with open(path) as f:
            color_wheel = json.load(f)
        color_wheel = [self.convert_color_value(i) for i in color_wheel]

        return np.array(color_wheel)
     
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

    def realtime_eyetracking(self,trial,wait_time,sampling_rate=.01):
        """Collect real time eyetracking data over a period of time

        Returns eyetracking data

        Parameters:
        wait_time-- How long in ms to collect data for
        sampling_rate-- How many ms between each sample
        """
        start_time = psychopy.core.getTime()

        eyes = None
        while psychopy.core.getTime() < start_time + wait_time:

            realtime_data = self.tracker.gaze_data

            reject,eyes = self.check_realtime_eyetracking(realtime_data)
#            reject=False
            if reject:
                self.rej_counter.append(trial)
                
                print(f'# of rejected trials this block: {len(self.rej_counter)}')
                    
                self.stop_eyetracking()
                self.display_eyemovement_feedback(eyes)
                return reject
            psychopy.core.wait(sampling_rate)
        
    def check_realtime_eyetracking(self,realtime_data):
        left_eye,right_eye = realtime_data
        if left_eye:
            lx,ly = left_eye
        if right_eye:
            rx,ry = right_eye        
        if (not left_eye) & (not right_eye):
            return False,None

        eyex = np.nanmean([lx,rx])
        eyey = np.nanmean([ly,ry])
        
        winx,winy = self.experiment_window.size/2
        
        eyex -= winx
        eyey -= winy
        eyes = np.array([eyex,eyey])

        limit_radius = psychopy.tools.monitorunittools.deg2pix(1.5,self.experiment_monitor)

        euclid_distance = np.linalg.norm(eyes-np.array([0,0])) 

        if euclid_distance > limit_radius:
            return True,(eyex,eyey)
        else:
            return False,None
            
    def display_eyemovement_feedback(self,eyes):

        psychopy.visual.TextStim(win=self.experiment_window,text='Eye Movement Detected.\nPress s to continue.',pos = [0,1], color = [1,-1,-1]).draw()
        psychopy.visual.TextStim(self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        if eyes is None:
            eyes = (0,0)
        psychopy.visual.Circle(win=self.experiment_window,radius=16,pos=eyes,fillColor='red',units='pix').draw()
        
        self.experiment_window.flip()
        psychopy.event.waitKeys(keyList=['space'])

    def handle_rejection(self,reject):
        self.rejection_tracker = np.roll(self.rejection_tracker,1)
        self.rejection_tracker[0] = reject
        
        if np.sum(self.rejection_tracker) == 5:
            self.rejection_tracker = np.zeros(5)
            self.display_text_screen(text='Rejected 5 in row\n\nContinue?',keyList = ['y'],bg_color=[0, 0, 255],text_color=[255,255,255])

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
        
    def send_synced_event(self, code, keyword = "SYNC"):
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
   
    def make_run(self, practice = False):

        # create counterbalanced trials
        condition = self.conditions
        alt = [0,1] # doesn't affect conditions, but helps with counterbalancing
        loc_1 = [0,1,2,3,4]
        loc_2 = [0,1,2,3,4]

        t = len(alt) * len(condition)
        conditiont = np.tile(condition,t//len(condition))
        altt = np.repeat(alt,t//len(alt))

        conditiont = np.tile(conditiont,len(loc_1))
        altt = np.tile(altt,len(loc_1))
        loc_1t = np.repeat(loc_1,t)

        t = len(loc_1t)
        conditiont = np.tile(conditiont,len(loc_2))
        altt = np.tile(altt,len(loc_2))
        loc_1t = np.tile(loc_1t,len(loc_2))
        loc_2t = np.repeat(loc_2,t)
        if practice is False:
            blocks = self.make_blocks(conditiont)
        else:
            blocks = self.make_practice_blocks(conditiont)

        run = []
        for iblock,block in enumerate(blocks):
            for itrial,trial in enumerate(block):
                run.append(self.make_trial(iblock,itrial,conditiont[trial],loc_1t[trial],loc_2t[trial]))
        
        return run

    def make_makeup_block(self, block_num, trial_num):

        run = []
        for trial in self.rej_counter:

            condition,loc1,loc2 = trial['condition'], trial['quints'][0], trial['quints'][1]
            run.append(self.make_trial(block_num, trial_num, condition, loc1, loc2))
            trial_num += 1

        self.rej_counter = []

        return run

    def make_practice_blocks(self, condition):
        
        int_idx = np.arange(len(condition))
        idx_ss2 = condition =='ss2'
        ss2_trials = np.random.permutation(int_idx[idx_ss2])

        blocks = []
        for cond in self.conditions[:3]:

            idx = condition == cond
            other_trials = np.random.permutation(int_idx[idx])
            block_trials = np.random.permutation(np.concatenate([ss2_trials[:5],other_trials[:5]]))
            blocks.append(block_trials)
        return blocks
    
    def make_blocks(self, condition):

        self.iblock = 0

        idx_ss2 = condition == 'ss2'
        int_idx = np.arange(len(condition))

        ss2a,ss2b,ss2c = np.split(np.random.permutation(int_idx[idx_ss2]),[16,32])

        # counterbalance order of blocks
        sub_shift = (int(self.experiment_info['Subject Number']) + int(self.experiment_info['Run'])) % 3 
        block_conditions = np.roll(self.conditions[:3],sub_shift)

        blocks = []
        for ss2_trials,cond in zip([ss2a,ss2b,ss2c],block_conditions):
            idx = condition == cond
            block_trials = np.random.permutation(np.concatenate([int_idx[idx],ss2_trials]))
            if self.number_of_trials_per_block is not None:
                block_trials = block_trials[:number_of_trials_per_block]
            blocks.append(block_trials)
        
        return blocks
        
    def make_trial(self, block_num,trial_num,condition,loc_1,loc_2):
        """Makes a single trial.

        Returns a dictionary of attributes about trial.
        """

        condition_dict = self.conditions_dict[condition]
        
        if condition == 'ss1_vdis':
            stim_color_idx = self.generate_color_indexes(1)
        else:
            stim_color_idx = self.generate_color_indexes(2)
        stim_colors = [self.color_wheel[i] for i in stim_color_idx]

        if condition == 'ss1_vdis':
            stim_colors.append(self.grey)

        wheel_rotation = random.randint(0, 359)

        locs = self.generate_locations((loc_1,loc_2))

        test_loc = locs[0]

        trial = {
            'run_num': self.experiment_info['Run'], 
            'trial_num': trial_num,
            'block_num': block_num,
            'code': int(condition_dict['code']),
            'set_size': condition_dict['set_size'],
            'condition': condition_dict['condition'],
            'num_distractors': condition_dict['num_distractors'],
            'locations': locs,
            'test_location': test_loc,
            'quints': [int(loc_1),int(loc_2)],
            'test_item': stim_colors[0],
            'colors': stim_colors,
            'color_indexes': stim_color_idx,
            'wheel_rotation': wheel_rotation
        }

        return trial

    def _check_dist(self, attempt, colors):
        """
        Checks if a color attempt statistfies the distance condition.
        Parameters:
            attempt -- The color index to be checked.
            colors -- The list of color indexes to be checked against.
        """
        for c in colors:
            raw_dist = abs(c - attempt)
            dist = min(raw_dist, 360 - raw_dist)

            if dist < self.min_color_dist:
                return False

        return True

    def generate_color_indexes(self, set_size):
        """
        Generates colors for a trial given the minimum distance.
        Parameters:
            set_size -- The number of colors to generate.
        """
        colors = []

        while len(colors) < set_size:
            attempt = random.randint(0, 359)
            if self._check_dist(attempt, colors):
                colors.append(attempt)

        return colors    

    def _too_close(self, attempt, locs):
        """Checks that an attempted location is valid.

        This method is used by generate_locations to ensure the min_distance condition is followed.

        Parameters:
        attempt -- A list of two values (x,y) in visual angle.
        locs -- A list of previous successful attempts to be checked.
        """
        
        # Too close to center
        if np.linalg.norm(np.array(attempt)) < self.min_distance/1.5:
            return True  
            
        for loc in locs:
            # Too close to another square
            if np.linalg.norm(np.array(attempt) - np.array(loc)) < self.min_distance:
                return True  

        return False

    def generate_locations(self, quints):

        locs = []
        counter = 0

        iloc = 0
        while len(locs) < 2:

            quint = quints[iloc]
            quint_bounds = self.quints_dict[quint]

            if quint == 4:
                a = list(range(0,quint_bounds[1]))
                b = list(range(quint_bounds[0],360))
                r = a + b
            else:
                r = list(range(quint_bounds[0],quint_bounds[1]))
            
            angle = np.random.choice(r)
            a = angle*math.pi / 180
            attempt = (self.distance_from_fix*math.cos(a),self.distance_from_fix*math.sin(a))

            if self._too_close(attempt, locs):
                continue
            else:
                locs.append(attempt)
                iloc += 1

            counter += 1

        return locs

    def display_start_block_screen(self):

        self.display_text_screen(text='Press space to begin the next block.',keyList=['space'],bg_color=[200,200,200])

    def display_fixation(self, wait_time = None, text = None, keyList = None, realtime_eyetracking = False, trial = None):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        text -- Str that displays above fixation cross. 
        keyList -- If keyList is given, will wait until key press
        trial -- Trial object needed for realtime eyetracking functionality.
        real_time_eyetracking -- Bool for if you want to do realtime eyetracking or not
        """
        
        if text:
            psychopy.visual.TextStim(win=self.experiment_window,text=text,pos = [0,1], color = [1,-1,-1]).draw()

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        self.experiment_window.flip()

        if realtime_eyetracking:
            reject = self.realtime_eyetracking(trial,wait_time=wait_time)
            return reject    
        else:
            if keyList:
                resp = psychopy.event.waitKeys(maxWait=wait_time,keyList=keyList)
                if resp == ['p']:
                    self.display_text_screen(text='Paused',keyList = ['space'])
                    self.display_fixation(wait_time=1)
                elif resp == ['o']:
                    self.tracker.calibrate()
                    self.display_fixation(wait_time=1)
                elif resp == ['escape']:
                    resp = self.display_text_screen(text='Are you sure you want to exit?',keyList = ['y','n'])
                    if resp == ['y']:
                        self.tracker.transfer_edf()
                        self.quit_experiment()
                    else:
                        self.display_fixation(wait_time=1)
            else:
                psychopy.core.wait(wait_time)
                
    def draw_stim(self, trial):

        #draw stim
        for istim in range(trial['set_size']):
            self.target.fillColor = trial['colors'][istim]
            self.target.pos = trial['locations'][istim]
            self.target.draw()

        # draw distractor
        if trial['condition'] == 'ss1_sim':
            self.sim_distractor.fillColor = trial['colors'][1]
            self.sim_distractor.pos = trial['locations'][1]
            self.sim_distractor.draw()
        if trial['condition'] == 'ss1_dis' or trial['condition'] == 'ss1_vdis':
            self.dis_distractor.fillColor = trial['colors'][1]
            self.dis_distractor.pos = trial['locations'][1]
            self.dis_distractor.draw()
    
    def draw_trak(self,x=930, y=510):
        trak = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor = [1,1,1], 
            fillColorSpace='rgb', radius=20, pos = [x,y], units='pix'
        )
        
        trak.draw()

    def display_stimuli(self, trial, realtime_eyetracking=False):
        """Displays the stimuli. A helper function for self.run_trial.

        Parameters:
        locations -- A list of locations (list of x and y value) describing where the stimuli
            should be displayed.
        colors -- A list of colors describing what should be drawn at each coordinate.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        self.draw_stim(trial)

        self.draw_trak()
        self.send_synced_event(trial['code'])
        self.experiment_window.flip()

        if realtime_eyetracking:  
            reject = self.realtime_eyetracking(trial,wait_time=self.sample_time)
            return reject
        else:
            psychopy.core.wait(self.sample_time)

    def draw_color_wheels(self, coordinates, wheel_rotation):
        """
        Draws color wheels at stimuli locations with random rotation.
        Parameters:
            coordinates -- A list of (x, y) tuples
            wheel_rotations -- A list of 0:359 ints describing how much each wheel
                should be rotated.
        """
        mask = np.zeros([100, 1])
        mask[-30:] = 1

        rotated_wheel = np.roll(self.color_wheel, wheel_rotation, axis=0)
        tex = np.repeat(rotated_wheel[np.newaxis, :, :], 360, 0)

        self.color_wheel_stim.pos = coordinates
        self.color_wheel_stim.tex = tex

        self.color_wheel_stim.draw()

    def _calc_mouse_color(self, mouse_pos):
        """
        Calculates the color of the pixel the mouse is hovering over.
        Parameters:
            mouse_pos -- A position returned by mouse.getPos()
        """
        frame = np.array(self.experiment_window._getFrame())  # Uses psychopy internal function

        x_correction = self.experiment_window.size[0] / 2
        y_correction = self.experiment_window.size[1] / 2

        x = int(psychopy.tools.monitorunittools.deg2pix(mouse_pos[0], self.experiment_monitor) + x_correction)
        y = (self.experiment_window.size[1] -
             int(psychopy.tools.monitorunittools.deg2pix(mouse_pos[1], self.experiment_monitor) + y_correction))

        try:
            color = frame[y, x, :]
        except IndexError:
            color = None

        return color

    def _calc_mouse_position(self, coordinates, mouse_pos):
        """
        Determines which position is closest to the mouse in order to display the hover preview.
        Parameters:
            coordinates -- A list of (x, y) tuples
            mouse_pos -- A position returned by mouse.getPos()
        """
        dists = [np.linalg.norm(np.array(i) - np.array(mouse_pos) / 2) for i in coordinates]
        closest_dist = min(dists)

        if closest_dist < 4:
            return coordinates[np.argmin(dists)]
        else:
            return None

    def _response_loop(self, coordinates, wheel_rotations):
        """
        Handles the hover updating and response clicks
        Slightly slow due to how psychopy handles clicks, so a full click and hold is needed.
        Parameters:
            coordinates -- A list of (x, y) tuples
            wheel_rotations -- A list of 0:359 ints describing how much each wheel
                should be rotated.
        """
        temp_coordinates = copy.copy([coordinates])
        temp_rotations = copy.copy([wheel_rotations])

        self.mouse.clickReset()

        self.draw_color_wheels(temp_coordinates, temp_rotations)
        self.experiment_window.flip()

        while True:

            (lclick, _, _), (rt, _, _) = self.mouse.getPressed(getTime=True)

            mouse_pos = self.mouse.getPos()
            px_color = self._calc_mouse_color(mouse_pos)

            if px_color is not None and not np.array_equal(px_color, np.array([127, 127, 127])):
                preview_pos = self._calc_mouse_position(temp_coordinates, mouse_pos)

                if preview_pos:
                    if lclick:

                        resp_color = px_color
                        return resp_color, rt

                    else:

                        psychopy.visual.Circle(
                            self.experiment_window, radius=self.square_target_size[0] / 2, pos=preview_pos,
                            fillColor=self.convert_color_value(px_color), units='deg',
                            lineColor=None).draw()

            self.draw_color_wheels(temp_coordinates, temp_rotations)
            self.experiment_window.flip()

    def get_response(self, coordinates, wheel_rotations):
        """
        Manages getting responses for all color wheels.
        Parameters:
            coordinates -- A list of (x, y) tuples
            wheel_rotations -- A list of 0:359 ints describing how much each wheel
                should be rotated.
        """
        
        self.send_synced_event(3)

        if not self.mouse:
            self.mouse = psychopy.event.Mouse(visible=False, win=self.experiment_window)

        self.mouse.setVisible(1)
        psychopy.event.clearEvents()

        resp_color, rt = self._response_loop(coordinates, wheel_rotations)

        self.mouse.setVisible(0)

        return resp_color, rt

    def calculate_error(self, color_index, resp_color):
        """
        Calculates error in a response compared to the true color value.
        Parameters:
            color_index -- The index of the true color values (0:359).
            resp_color -- The rgb color that was selected.
        """
        
        row_index = np.where((np.array(self.color_wheel)==resp_color).all(axis=1))[0]

        if row_index.shape[0] < 1:
            return None  # if empty, return None

        raw_error = row_index[0] - color_index
        if raw_error >= -180 and raw_error <= 180:
            error = raw_error
        elif raw_error < -180:
            error = 360 + raw_error
        else:
            error = raw_error - 360
        
        return error,row_index

    def send_data(self, data):
        """Updates the experiment data with the information from the last trial.

        This function is seperated from run_trial to allow additional information to be added
        afterwards.

        Parameters:
        data -- A dict where keys exist in data_fields and values are to be saved.
        """
        self.update_experiment_data([data])

    def run_trial(self, trial, realtime_eyetracking=False):
        """Runs a single trial.

        Returns the data from the trial after getting a participant response.

        Parameters:
        trial -- The dictionary of information about a trial.
        block_num -- The number of the block in the experiment.
        trial_num -- The number of the trial within a block.
        """
        self.display_fixation(wait_time=np.random.randint(400,601)/1000,trial=trial,keyList=['p','escape','o'])
        self.start_eyetracking(block_num = trial['block_num'], trial_num = trial['trial_num']) 
        
        self.send_synced_event(1)
        reject = self.display_fixation(wait_time=self.iti_time,trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        reject = self.display_stimuli(
            trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        self.send_synced_event(2)
        reject = self.display_fixation(self.delay_time, trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        print(trial['quints'])
        print(trial['locations'])
        
        resp_color, rt = self.get_response(trial['locations'][0], trial['wheel_rotation'])
        color = self.convert_color_value(resp_color)
        error,resp_index = self.calculate_error(trial['color_indexes'][0], color)

        true_color = self.convert_color_value(trial['test_item'],deconvert=True)
        colors = [self.convert_color_value(n,deconvert=True) for n in trial['colors']]
        
        self.send_synced_event(4)
        self.stop_eyetracking()
        self.handle_rejection(0)
        
        data = {
            'Subject': self.experiment_info['Subject Number'],
            'Run': trial['run_num'],
            'Block': trial['block_num'],
            'Trial': trial['trial_num'],
            'Timestamp': psychopy.core.getAbsTime(),
            'Condition': trial['condition'],
            'SetSize': json.dumps(trial['set_size']),
            'RT': rt,
            'Colors': colors, 
            'ColorIndex':trial['color_indexes'],
            'TrueColor': true_color,
            'RespColor':resp_color,
            'RespIndex':resp_index[0],
            'Error':error,
            'Locations': json.dumps(trial['locations']),
            'TestLocation': json.dumps(trial['test_location']),
            'Quintants': json.dumps(trial['quints']),
            'NumDistractors': json.dumps(trial['num_distractors'])
        }

        print(f'{trial["block_num"]+1}, {trial["trial_num"]+1}')

        print(error)
        return data

    def run_makeup_block(self, block_num, trial_num):

        if len(self.rej_counter) > 10: #FIX

            makeup_block = self.make_makeup_block(block_num,trial_num)
            self.display_start_block_screen()

            for makeup_trial in makeup_block:
                print('makeup trial')

                if makeup_trial['trial_num'] % 5 == 0:
                    self.tracker.drift_correct()

                data = self.run_trial(makeup_trial, realtime_eyetracking = True)
                if data:
                    self.send_data(data)
        
        self.rej_counter = []

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
        data_filename = (f'{self.experiment_name}_{self.experiment_info["Subject Number"].zfill(2)}_{str(self.experiment_info["Run"])}')
        self.open_csv_data_file(data_filename)
        self.open_window(screen=0)
        self.display_text_screen('Loading...', wait_for_input=False)

        self.init_tracker()
        self.init_stim()

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction, keyList=['space'])
        
        self.show_eyetracking_instructions()

        """
        Practice
        """
        self.port = None
        prac = self.display_text_screen(text = f'Practice block?', keyList=['y','n'])
        
        while prac == ['y']: 
            
            run = self.make_run(practice=True)
            acc = []
            
            for trial in run:

                if trial['trial_num'] == 0:
                    self.display_start_block_screen() 
                
                data = self.run_trial(trial)      
                acc.append(abs(data['Error']))

            self.display_text_screen(
                text = f'Block Error: {round(np.nanmean(acc)),1}\n\n\n\nPress space to continue.',keyList=['space'])
        
            prac = self.display_text_screen(text = f'Practice block?', keyList=['y','n'])
            
        """
        Experiment
        """
        self.setup_eeg()
        run = self.make_run()
        self.tracker.calibrate()

        self.rejection_tracker = np.zeros(5)
        itrial = 0
        acc = []        

        for trial in run:

            if trial['trial_num'] == 0:
                
                # before starting next block, check if any makeup trials are needed
                self.run_makeup_block(trial['block_num']-1, itrial)
                
                itrial = 0
                self.display_text_screen(
                    text = f'Block Error: {round(np.nanmean(acc),1)}\n\n\n\nPress space to start block.',keyList=['space'])
                acc = []    
                self.tracker.calibrate()

            if trial['trial_num'] % 5 == 0:
                self.tracker.drift_correct()

            data = self.run_trial(trial, realtime_eyetracking=True)
            if data:
                self.send_data(data)
                acc.append(abs(data['Error']))

            itrial += 1

        self.run_makeup_block(trial['block_num'], itrial)

        self.save_data_to_csv()

        """
        End of Experiment
        """
        self.display_text_screen(
            'The run is now over.',
            bg_color=[0, 0, 255], text_color=[255, 255, 255])
        
        self.tracker.transfer_edf()
        self.quit_experiment()

# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = Discus(
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
