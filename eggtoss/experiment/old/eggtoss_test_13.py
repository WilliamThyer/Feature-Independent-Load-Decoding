from psychopy import visual, event, monitors
from psychopy.visual import ShapeStim, TextStim, Circle, Rect
import numpy as np
import math

monitor_name='Experiment Monitor' 
monitor_width=53
monitor_distance=70
monitor_px=[1920, 1080]

experiment_monitor = monitors.Monitor(
            monitor_name, width=monitor_width,
            distance=monitor_distance)
experiment_monitor.setSizePix(monitor_px)

win = visual.Window(monitor = experiment_monitor, fullscr = True, units='deg')

def make_default_dots(coherence=1,fieldShape='circle'):
    dots = visual.DotStim(
            fieldShape=fieldShape,fieldSize=(2,2),
            noiseDots='direction',signalDots='same',
            speed=.03, dotLife=10,
            dotSize=7.5, color='blue',
            coherence=coherence, dir=90,
            nDots=100,
            win=win)
    
    return dots

rand = lambda lower, upper: np.random.choice(np.arange(lower,upper), size=1)[0]

coherence_levels = [.9,.9,.2,.2]
field_shape = 'circle'
dist = 'circle'
colorset = ['blue','red','green','yellow','orange','teal']
set_size = 1

dots_a = make_default_dots()
dots_b = make_default_dots()
dots_c = make_default_dots()
dots_d = make_default_dots()
dots = [dots_a,dots_b,dots_c,dots_d]

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

    # item attributes
    coherence = np.random.choice(coherence_levels,size=4,replace=False)
    a,b,c,d = (rand(-6,-2), rand(-6,-2)),(rand(-6,-2), rand(2,6)),(rand(2,6), rand(2,6)),(rand(2,6), rand(-6,-2)) 
    locs = np.random.permutation([a,b,c,d])
    colors = np.random.choice(colorset,size=set_size,replace=False)
    colors = np.concatenate([colors,np.repeat('lightgrey',4-set_size)])
    # ori = np.random.choice([1],size=4,replace = True)
    ori_set = np.random.choice(np.arange(0,359),4,replace=False)
    # ori = np.random.choice(,size=4,replace = True)

    dots_a.fieldPos,dots_b.fieldPos,dots_c.fieldPos,dots_d.fieldPos = locs[0],locs[1],locs[2],locs[3]
    dots_a.color,dots_b.color,dots_c.color,dots_d.color = colors[0],colors[1],colors[2],colors[3]
    dots_a.coherence,dots_b.coherence,dots_c.coherence,dots_d.coherence = coherence[0],coherence[1],coherence[2],coherence[3]
    dots_a.fieldShape,dots_b.fieldShape,dots_c.fieldShape,dots_d.fieldShape = np.concatenate([np.repeat(field_shape,set_size),np.repeat(dist,4-set_size)])
    # dots_a.dir,dots_b.dir,dots_c.dir,dots_d.dir = dots_a.dir*ori[0],dots_b.dir*ori[1],dots_c.dir*ori[2],dots_d.dir*ori[3]
    dots_a.dir,dots_b.dir,dots_c.dir,dots_d.dir = ori_set

    for n in range(50):
        dots_a.draw()
        dots_b.draw()
        dots_c.draw()
        dots_d.draw()
        
        win.flip() 
    
    if set_size == 1:
        set_size = 2
    else:
        set_size = 1

    resp = event.waitKeys(keyList=['space','escape','c','s','b'])

win.close()
quit()