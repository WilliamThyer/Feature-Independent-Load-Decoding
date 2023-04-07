from psychopy import visual, event, monitors
from psychopy.visual import ShapeStim, TextStim, Circle, Rect
import numpy as np
import math
import json
import template

print('starting')

monitor_name='Experiment Monitor' 
monitor_width=53
monitor_distance=70
monitor_px=[1920, 1080]

experiment_monitor = monitors.Monitor(
            monitor_name, width=monitor_width,
            distance=monitor_distance)
experiment_monitor.setSizePix(monitor_px)

win = visual.Window(monitor = experiment_monitor, fullscr = True, units='deg')

with open('colors.json') as f:
    color_wheel = json.load(f)

color_wheel = [template.convert_color_value(i) for i in color_wheel]

resp = ['space']
while resp[0] == 'space':

    wheel_rotation = np.random.choice(range(360))
    mask = np.zeros([100, 1])
    mask[-30:] = 1

    rotated_wheel = np.roll(color_wheel, wheel_rotation, axis=0)
    print(rotated_wheel.shape)
    tex = np.repeat(rotated_wheel[np.newaxis, :, :], 360, 0)
    print(tex.shape)
    coordinates = (1,1)

    rad = visual.RadialStim(
        win, tex=tex, mask=mask, pos=coordinates, angularRes=256,
        angularCycles=1, interpolate=False, size=2)

    rad.draw()

    win.flip() 
    resp = event.waitKeys(keyList=['space','escape'])
