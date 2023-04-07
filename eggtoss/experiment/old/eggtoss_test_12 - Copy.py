from psychopy import visual, event, monitors
from psychopy.visual import ShapeStim, TextStim, Circle, Rect
import numpy as np
import json 
import time

monitor_name='Experiment Monitor' 
monitor_width=53
monitor_distance=70
monitor_px=[1920, 1080]

min_color_dist = 25

#Color Setup
def color_convert(color):
    return [round(((n/127.5)-1), 2) for n in color]

color_array_idx = [0,1,2,3,4,5,6]
color_table = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0]
])

rgb_table = []
for colorx in color_array_idx:
    rgb_table.append(color_convert(color_table[colorx]))
rgb_table = np.array(rgb_table)
grey = np.array(color_convert([166,166,166]))

experiment_monitor = monitors.Monitor(
            monitor_name, width=monitor_width,
            distance=monitor_distance)
experiment_monitor.setSizePix(monitor_px)

win = visual.Window(monitor = experiment_monitor, fullscr = True, units='deg')

def make_default_dots(coherence=1,fieldShape='circle'):
    dots = visual.DotStim(
            fieldShape=fieldShape,fieldSize=(3,3),
            noiseDots='direction',signalDots='same',
            speed=.07, dotLife=4,
            dotSize=7.5, color=[.700,.700,.700],
            coherence=coherence, dir=90,
            nDots=100,
            win=win)
    
    return dots 

fixation_cross = visual.TextStim(win, text='+', height=1, color=[-1,-1,-1], pos=(0,0))

rand = lambda lower, upper: np.random.choice(np.arange(lower,upper), size=1)[0]

coherence_levels = [1,1,.4,.4]
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
        a,b,c = (rand(-3,-1.5), rand(-3,-1.5)),(rand(-3,-1.5), rand(1.5,3)),(rand(1.5,3), rand(1.5,3))
        locs = np.random.permutation([a,b,c])
        colors = np.random.choice(color_array_idx,size=set_size,replace=False)
        colors = rgb_table[colors]
        colors = np.concatenate([colors,np.tile(grey[:,np.newaxis],3-set_size).T])
        ori_set = np.random.choice(np.arange(0,359),3,replace=False)

        dots_a.fieldPos,dots_b.fieldPos,dots_c.fieldPos = locs[0],locs[1],locs[2]
        dots_a.color,dots_b.color,dots_c.color = colors[0],colors[1],colors[2]
        dots_a.coherence,dots_b.coherence,dots_c.coherence = coherence[0],coherence[1],coherence[2]
        dots_a.fieldShape,dots_b.fieldShape,dots_c.fieldShape = np.concatenate([np.repeat(field_shape,set_size),np.repeat(dist,3-set_size)])
        dots_a.dir,dots_b.dir,dots_c.dir = ori_set
    
    start = time.time()
    # for n in range(50):
    while time.time() - start < .75:
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