from psychopy import visual, event, monitors
from psychopy.visual import ShapeStim, TextStim, Circle, Rect
import numpy as np
import json 

monitor_name='Experiment Monitor' 
monitor_width=53
monitor_distance=70
monitor_px=[1920, 1080]

min_color_dist = 25

def convert_color_value(color, deconvert=False):
    """Converts a list of 3 values from 0 to 255 to -1 to 1.

    Parameters:
    color -- A list of 3 ints between 0 and 255 to be converted.
    """

    if deconvert is True:
        return [round((((n - -1) * 255) / 2) + 0,1) for n in color]
    else:
        return [round((((n - 0) * 2) / 255) + -1,3) for n in color]

def load_color_wheel(path):

    with open(path) as f:
        color_wheel = json.load(f)
    color_wheel = [convert_color_value(i) for i in color_wheel]

    return np.array(color_wheel)

color_wheel = load_color_wheel('colors.json')
color_idx = np.arange(0,len(color_wheel))

experiment_monitor = monitors.Monitor(
            monitor_name, width=monitor_width,
            distance=monitor_distance)
experiment_monitor.setSizePix(monitor_px)

win = visual.Window(monitor = experiment_monitor, fullscr = True, units='deg')

def make_default_dots(coherence=1,fieldShape='circle'):
    dots = visual.DotStim(
            fieldShape=fieldShape,fieldSize=(3,3),
            noiseDots='direction',signalDots='same',
            speed=.04, dotLife=10,
            dotSize=7.5, color=[.700,.700,.700],
            coherence=coherence, dir=90,
            nDots=100,
            win=win)
    
    return dots

fixation_cross = visual.TextStim(win, text='+', height=1, color=[-1,-1,-1],pos=(0,0))

rand = lambda lower, upper: np.random.choice(np.arange(lower,upper), size=1)[0]

coherence_levels = [.9,.9,0,0]
field_shape = 'circle'
dist = 'circle'
colorset = ['blue','red','green','yellow','orange','teal']
set_size = 1

dots_a = make_default_dots()
dots_b = make_default_dots()
dots_c = make_default_dots()
dots = [dots_a,dots_b,dots_c]

repeat = False
resp = ['space']
while resp[0] != 'escape':
    
    if resp[0] == 'c':
        field_shape = 'circle'
        dist = 'square'
        
    if resp[0] == 's':
        field_shape = 'square'
        dist = 'circle'
    
    if resp[0] == 'b':
        field_shape = 'circle'
        dist = 'circle'
    
    if resp[0] == 'r':
        repeat = True
    else:
        repeat = False

    if repeat is False:
        # item attributes
        coherence = np.random.choice(coherence_levels,size=3,replace=False)
        a,b,c = (rand(-5,-2), rand(-5,-2)),(rand(-5,-2), rand(2,5)),(rand(2,5), rand(2,5))
        locs = np.random.permutation([a,b,c])
        colors = np.random.choice(color_idx,size=set_size,replace=False)
        colors = color_wheel[colors]
        colors = np.concatenate([colors,np.tile(np.array([.700,.700,.700])[:,np.newaxis],3-set_size).T])
        print(colors)
    # np.tile(np.array([200,200,200])[:,np.newaxis],2)
        # ori = np.random.choice([1],size=4,replace = True)
        ori_set = np.random.choice(np.arange(0,359),3,replace=False)
        # ori = np.random.choice(,size=4,replace = True)

        dots_a.fieldPos,dots_b.fieldPos,dots_c.fieldPos = locs[0],locs[1],locs[2]
        dots_a.color,dots_b.color,dots_c.color = colors[0],colors[1],colors[2]
        dots_a.coherence,dots_b.coherence,dots_c.coherence = coherence[0],coherence[1],coherence[2]
        dots_a.fieldShape,dots_b.fieldShape,dots_c.fieldShape = np.concatenate([np.repeat(field_shape,set_size),np.repeat(dist,3-set_size)])
        # dots_a.dir,dots_b.dir,dots_c.dir,dots_d.dir = dots_a.dir*ori[0],dots_b.dir*ori[1],dots_c.dir*ori[2],dots_d.dir*ori[3]
        dots_a.dir,dots_b.dir,dots_c.dir = ori_set

    
    for n in range(50):
        dots_a.draw()
        dots_b.draw()
        dots_c.draw()
        fixation_cross.draw()
        
        win.flip() 
    
    if set_size == 1:
        set_size = 2
    else:
        set_size = 1

    resp = event.waitKeys(keyList=['space','escape','c','s','b','r'])

win.close()
quit()