"""Basic experiment class that is designed to be extended.

Author - Colin Quirk (cquirk@uchicago.edu)

Repo: https://github.com/colinquirk/templateexperiments

This class provides basic utility functions that are needed by all
experiments. Specific experiment classes should inherit and extend/overwrite as needed.

Note for other experimenters -- My experiments all inherit from this class,
so changes in these functions may result in unexpected changes elsewhere. If
possible, changes to experiments should be made in the specific experiment
class by overwriting template experiment methods. Ideally, the only changes
that should be made to these classes are those that would need to be made for
every experiment of mine (e.g. to correct for system differences). Even those
types of changes may have unintended consequences so please be careful! If you
need help using this module, have requests or improvements, or found this code
useful please let me know through email or GitHub.

Functions:
convert_color_value -- Converts a list of 3 values from 0 to 255 to -1 to 1.

Classes:
BaseExperiment -- All experiments inherit from BaseExperiment. Provides basic
    functionality needed by all experiments.
    See 'print templateexperiments.BaseExperiment.__doc__' for simple class
    docs or help(templateexperiments.BaseExperiment) for everything.
"""

import json
import os
import pickle
import sys

import psychopy.monitors
import psychopy.visual
import psychopy.gui
import psychopy.core
import psychopy.event


# Convenience
def convert_color_value(color):
    """Converts a list of 3 values from 0 to 255 to -1 to 1.

    Parameters:
    color -- A list of 3 ints between 0 and 255 to be converted.
    """

    return [round(((n/127.5)-1), 2) for n in color]


class BaseExperiment:
    """Basic experiment class providing functionality in all experiments

    Parameters:
    bg_color -- list of 3 values (0-255) defining the background color
    data_fields -- list of strings defining data fields
    experiment_name -- string defining the experiment title
    monitor_distance -- int describing participant distance from monitor in cm
    monitor_name -- name of the monitor to be used
    monitor_px -- list containing monitor resolution (x,y)
    monitor_width -- int describing length of display monitor in cm

    Methods:
    display_text_screen -- draws a string centered on the screen.
    get_experiment_info_from_dialog -- gets subject info from a dialog box.
    open_csv_data_file -- opens a csv data file and writes the header.
    open_window -- open a psychopy window.
    quit_experiment -- ends the experiment.
    save_data_to_csv -- append new entries in experiment_data to csv data file.
    save_experiment_info -- write the info from the dialog box to a text file.
    save_experiment_pickle -- save a pickle so crashes can be recovered from.
    update_experiment_data -- extends any new data to the experiment_data list.
    """

    def __init__(self, experiment_name, data_fields, bg_color=[128, 128, 128],
                 monitor_name='Experiment Monitor', monitor_width=53,
                 monitor_distance=70, monitor_px=[1920, 1080], **kwargs):
        """Creates a new BaseExperiment object.

        Parameters:
        bg_color -- A list of 3 values between 0 and 255 defining the
            background color.
        data_fields -- list of strings containing the data fields to be stored
        experiment_name -- A string for the experiment title that also defines
            the filename the experiment info from the dialog box is saved to.
        monitor_distance -- An int describing the distance the participant sits
            from the monitor in cm (default 70).
        monitor_name -- The name of the monitor to be used
            Psychopy will search for the provided name to see if it was defined
            in monitor center. If it is not defined, a temporary monitor will
            be created.
        monitor_px -- A list containing the resolution of the monitor (x,y).
        monitor_width -- An int describing the length of the display monitor
            in cm (default 53).
        """

        self.experiment_name = experiment_name
        self.data_fields = data_fields
        self.bg_color = convert_color_value(bg_color)
        self.monitor_name = monitor_name
        self.monitor_width = monitor_width
        self.monitor_distance = monitor_distance
        self.monitor_px = monitor_px

        self.experiment_data = []
        self.experiment_data_filename = None
        self.data_lines_written = 0
        self.experiment_info = {}
        self.experiment_window = None

        self.overwrite_ok = None

        self.experiment_monitor = psychopy.monitors.Monitor(
            self.monitor_name, width=self.monitor_width,
            distance=self.monitor_distance)
        self.experiment_monitor.setSizePix(monitor_px)

        vars(self).update(kwargs)  # Add anything else you want

    @staticmethod
    def _confirm_overwrite(screen=0):
        """Private, static method that shows a dialog asking if a file can be
        overwritten.

        Returns a bool describing if the file should be overwritten.

        Parameters:
        screen -- an int describing the screen you want the dialog to appear on
        """

        overwrite_dlg = psychopy.gui.Dlg(
            'Overwrite?', labelButtonOK='Overwrite',
            labelButtonCancel='New File', screen=screen)
        overwrite_dlg.addText('File already exists. Overwrite?')
        overwrite_dlg.show()

        return overwrite_dlg.OK

    def get_experiment_info_from_dialog(self, additional_fields_dict=None, screen=0):
        """Gets subject info from dialog box.

        Parameters:
        additional_fields_dict -- An optional dictionary containing more
            fields for the dialog box and output dictionary.
        screen -- an int describing the screen you want the dialog to appear on
        """

        self.experiment_info = {'Subject Number': '0',
                                'Age': '0',
                                'Experimenter Initials': 'CQ',
                                'Unique Subject Identifier': '000000'
                                }

        if additional_fields_dict is not None:
            self.experiment_info.update(additional_fields_dict)

        # Modifies experiment_info dict directly
        exp_info = psychopy.gui.DlgFromDict(
            self.experiment_info, title=self.experiment_name,
            order=['Subject Number',
                   'Age',
                   'Experimenter Initials',
                   'Unique Subject Identifier'
                   ],
            tip={'Unique Subject Identifier': 'From the cronus log'},
            screen=screen
        )

        return exp_info.OK

    def save_experiment_info(self, filename=None):
        """Writes the info from the dialog box to a json file.

        Parameters:
        filename -- a string defining the filename with no extension
        """

        ext = '.json'

        if filename is None:
            filename = (self.experiment_name + '_' +
                        self.experiment_info['Subject Number'].zfill(3) +
                        '_info')
        elif filename[-5:] == ext:
            filename = filename[:-5]

        if os.path.isfile(filename + ext):
            if self.overwrite_ok is None:
                self.overwrite_ok = self._confirm_overwrite()
            if not self.overwrite_ok:
                # If the file exists make a new filename
                i = 1
                new_filename = filename + '(' + str(i) + ')'
                while os.path.isfile(new_filename + ext):
                    i += 1
                    new_filename = filename + '(' + str(i) + ')'
                filename = new_filename

        filename = filename + ext

        with open(filename, 'w') as info_file:
            info_file.write(json.dumps(self.experiment_info))

    def open_csv_data_file(self, data_filename=None):
        """Opens the csv file and writes the header.

        Parameters:
        data_filename -- name of the csv file with no extension
            (defaults to experimentname_subjectnumber).
        """

        if data_filename is None:
            data_filename = (self.experiment_name + '_' +
                             self.experiment_info['Subject Number'].zfill(3))
        elif data_filename[-4:] == '.csv':
            data_filename = data_filename[:-4]

        if os.path.isfile(data_filename + '.csv'):
            if self.overwrite_ok is None:
                self.overwrite_ok = self._confirm_overwrite()
            if not self.overwrite_ok:
                # If the file exists and we can't overwrite make a new filename
                i = 1
                new_filename = data_filename + '(' + str(i) + ')'
                while os.path.isfile(new_filename + '.csv'):
                    i += 1
                    new_filename = data_filename + '(' + str(i) + ')'
                data_filename = new_filename

        self.experiment_data_filename = data_filename + '.csv'

        # Write the header
        with open(self.experiment_data_filename, 'w+') as data_file:
            for field in self.data_fields:
                data_file.write('"')
                data_file.write(field)
                data_file.write('"')
                if field != self.data_fields[-1]:
                    data_file.write(',')
            data_file.write('\n')

    def update_experiment_data(self, new_data):
        """Extends any new data to the experiment_data list.

        Parameters:
        new_data -- A list of dictionaries that are extended to
            experiment_data. Only keys that are included in data_fields should
            be included, as only those will be written in save_data_to_csv()
        """
        if not isinstance(new_data, list):
            raise TypeError('Experiment data must be type list.')

        self.experiment_data.extend(new_data)

    def save_data_to_csv(self):
        """Opens the data file and appends new entries in experiment_data.

        Only appends lines (tracked by data_lines_written) that have not yet
        been written to the csv.

        Update the experiment data to be written with update_experiment_data.
        """

        with open(self.experiment_data_filename, 'a') as data_file:
            for trial in range(
                    self.data_lines_written, len(self.experiment_data)):
                for field in self.data_fields:
                    data_file.write('"')
                    try:
                        data_file.write(
                            str(self.experiment_data[trial][field]))
                    except KeyError:
                        data_file.write('NA')
                    data_file.write('"')
                    if field != self.data_fields[-1]:
                        data_file.write(',')
                data_file.write('\n')

        self.data_lines_written = len(self.experiment_data)

    def save_experiment_pickle(self, additional_fields_dict=None):
        """Saves the pickle containing the experiment data so that a crash can
        be recovered from.

        This method uses dict.update() so if any keys in the
        additional_fields_dict are in the default dictionary the new values
        will be used.

        Parameters:
        additional_fields_dict -- An optional dictionary that updates the
            dictionary that is saved in the pickle file.
        """

        pickle_dict = {
            'experiment_name': self.experiment_name,
            'data_fields': self.data_fields,
            'bg_color': self.bg_color,
            'monitor_name': self.monitor_name,
            'monitor_width': self.monitor_width,
            'monitor_distance': self.monitor_distance,
            'experiment_data': self.experiment_data,
            'experiment_data_filename': self.experiment_data_filename,
            'data_lines_written': self.data_lines_written,
            'experiment_info': self.experiment_info,
        }

        if additional_fields_dict is not None:
            pickle_dict.update(additional_fields_dict)

        pickle_filename = (self.experiment_name + '_' +
                           self.experiment_info['Subject Number'].zfill(3) + '.pickle')

        with open(pickle_filename, 'wb+') as pickle_file:
            pickle.dump(pickle_dict, pickle_file)

    def open_window(self, **kwargs):
        """Opens the psychopy window.

        Additional keyword arguments are sent to psychopy.visual.Window().
        """
        self.experiment_window = psychopy.visual.Window(
            monitor=self.experiment_monitor, fullscr=True, color=self.bg_color,
            colorSpace='rgb', units='deg', allowGUI=False, **kwargs)

    def display_text_screen(
            self, text='', text_color=[0, 0, 0], text_height=36,
            bg_color=None, wait_for_input=True, keyList=None, **kwargs):
        """Takes a string as input and draws it centered on the screen.

        Allows for simple writing of text to a screen with a background color
        other than the normal one. Switches back to the default background
        color after any keyboard input.

        This works by drawing a rect on top of the background
        that fills the whole screen with the selected color.

        Parameters:
        text -- A string containing the text to be displayed.
        text_color -- A list of 3 values between 0 and 255
            (default is [0, 0, 0]).
        text_height --- An int that defines the height of the text in pix.
        bg_color -- A list of 3 values between 0 and 255 (default is default
            background color).
        wait_for_input -- Bool that defines whether the screen will wait for
            keyboard input before continuing. If waiting for keys, a .5 second
            buffer is added to prevent accidental advancing.

        Additional keyword arguments are sent to psychopy.visual.TextStim().
        """

        if bg_color is None:
            bg_color = self.bg_color
        else:
            bg_color = convert_color_value(bg_color)

        backgroundRect = psychopy.visual.Rect(
            self.experiment_window, fillColor=bg_color, units='norm', width=2,
            height=2)

        text_color = convert_color_value(text_color)

        textObject = psychopy.visual.TextStim(
            self.experiment_window, text=text, color=text_color, units='pix',
            height=text_height, alignHoriz='center', alignVert='center',
            wrapWidth=round(.8*self.experiment_window.size[0]), **kwargs)

        backgroundRect.draw()
        textObject.draw()
        self.experiment_window.flip()

        keys = None

        if wait_for_input:
            psychopy.core.wait(.2)  # Prevents accidental key presses
            keys = psychopy.event.waitKeys(keyList=keyList)
            self.experiment_window.flip()

        return keys

    def quit_experiment(self):
        """Completes anything that must occur when the experiment ends."""
        if self.experiment_window:
            self.experiment_window.close()
        print('The experiment has ended.')
        sys.exit(0)


class EyeTrackingEEGExperiment(BaseExperiment):
    def __init__(self, *args, tracker=None, eeg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.eeg = eeg

    def send_synced_event(self, code, keyword="SYNC", end_eeg_event=False):
        if keyword is None:
            message = str(code)
        else:
            message = keyword + ' ' + str(code)

        self.eeg.start_event(code)
        self.tracker.send_message(message)
        if end_eeg_event:
            self.eeg.end_eeg_event()
