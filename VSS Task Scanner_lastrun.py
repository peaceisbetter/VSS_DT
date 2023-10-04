#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on October 04, 2023, at 11:19
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2023.2.2')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '2'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'VSS Task'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\jackpmanning\\OneDrive - Texas A&M University\\Documents\\Projects\\F31 Application\\Psycho Py Task\\Visual Set Shifting\\VSS Task Scanner_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    # Make folder to store recordings from mic
    micRecFolder = filename + '_mic_recorded'
    if not os.path.isdir(micRecFolder):
        os.mkdir(micRecFolder)
    
    # --- Initialize components for Routine "Instructions" ---
    instruct = visual.TextStim(win=win, name='instruct',
        text='If background is Blue say the letter, if it is Orange say the Number',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "PleaseWait" ---
    MRI_input_trigger = keyboard.Keyboard()
    PleaseWaitToStart = visual.TextStim(win=win, name='PleaseWaitToStart',
        text='Please Wait',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Waittime" ---
    Fixation = visual.TextStim(win=win, name='Fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # create a microphone object for device: Microphone Array (Realtek(R) Audio)
    microphoneArrayRealtekRAudio = sound.microphone.Microphone(
        device='5', channels=None, 
        sampleRateHz=48000, maxRecordingSize=24000.0
    )
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "trial" ---
    vsstask = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.3,
         size=(2.5, 2.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='vsstask',
         depth=0, autoLog=False,
    )
    # link mic to device object
    mic = microphoneArrayRealtekRAudio
    
    # --- Initialize components for Routine "halfsecfix" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "BlockRest" ---
    text = visual.TextStim(win=win, name='text',
        text='Please Wait\n\nDo Not Move',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Instructions.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    InstructionsComponents = [instruct, key_resp]
    for thisComponent in InstructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruct* updates
        
        # if instruct is starting this frame...
        if instruct.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruct.frameNStart = frameN  # exact frame index
            instruct.tStart = t  # local t and not account for scr refresh
            instruct.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruct, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruct.started')
            # update status
            instruct.status = STARTED
            instruct.setAutoDraw(True)
        
        # if instruct is active this frame...
        if instruct.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in InstructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instructions" ---
    for thisComponent in InstructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Run_repeat = data.TrialHandler(nReps=3.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='Run_repeat')
    thisExp.addLoop(Run_repeat)  # add the loop to the experiment
    thisRun_repeat = Run_repeat.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun_repeat.rgb)
    if thisRun_repeat != None:
        for paramName in thisRun_repeat:
            globals()[paramName] = thisRun_repeat[paramName]
    
    for thisRun_repeat in Run_repeat:
        currentLoop = Run_repeat
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRun_repeat.rgb)
        if thisRun_repeat != None:
            for paramName in thisRun_repeat:
                globals()[paramName] = thisRun_repeat[paramName]
        
        # --- Prepare to start Routine "PleaseWait" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('PleaseWait.started', globalClock.getTime())
        MRI_input_trigger.keys = []
        MRI_input_trigger.rt = []
        _MRI_input_trigger_allKeys = []
        # keep track of which components have finished
        PleaseWaitComponents = [MRI_input_trigger, PleaseWaitToStart]
        for thisComponent in PleaseWaitComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "PleaseWait" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *MRI_input_trigger* updates
            waitOnFlip = False
            
            # if MRI_input_trigger is starting this frame...
            if MRI_input_trigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MRI_input_trigger.frameNStart = frameN  # exact frame index
                MRI_input_trigger.tStart = t  # local t and not account for scr refresh
                MRI_input_trigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MRI_input_trigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'MRI_input_trigger.started')
                # update status
                MRI_input_trigger.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(MRI_input_trigger.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(MRI_input_trigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if MRI_input_trigger.status == STARTED and not waitOnFlip:
                theseKeys = MRI_input_trigger.getKeys(keyList=['t', 'T'], ignoreKeys=["escape"], waitRelease=False)
                _MRI_input_trigger_allKeys.extend(theseKeys)
                if len(_MRI_input_trigger_allKeys):
                    MRI_input_trigger.keys = _MRI_input_trigger_allKeys[-1].name  # just the last key pressed
                    MRI_input_trigger.rt = _MRI_input_trigger_allKeys[-1].rt
                    MRI_input_trigger.duration = _MRI_input_trigger_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *PleaseWaitToStart* updates
            
            # if PleaseWaitToStart is starting this frame...
            if PleaseWaitToStart.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                PleaseWaitToStart.frameNStart = frameN  # exact frame index
                PleaseWaitToStart.tStart = t  # local t and not account for scr refresh
                PleaseWaitToStart.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(PleaseWaitToStart, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'PleaseWaitToStart.started')
                # update status
                PleaseWaitToStart.status = STARTED
                PleaseWaitToStart.setAutoDraw(True)
            
            # if PleaseWaitToStart is active this frame...
            if PleaseWaitToStart.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in PleaseWaitComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "PleaseWait" ---
        for thisComponent in PleaseWaitComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('PleaseWait.stopped', globalClock.getTime())
        # check responses
        if MRI_input_trigger.keys in ['', [], None]:  # No response was made
            MRI_input_trigger.keys = None
        Run_repeat.addData('MRI_input_trigger.keys',MRI_input_trigger.keys)
        if MRI_input_trigger.keys != None:  # we had a response
            Run_repeat.addData('MRI_input_trigger.rt', MRI_input_trigger.rt)
            Run_repeat.addData('MRI_input_trigger.duration', MRI_input_trigger.duration)
        # the Routine "PleaseWait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Waittime" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Waittime.started', globalClock.getTime())
        # keep track of which components have finished
        WaittimeComponents = [Fixation]
        for thisComponent in WaittimeComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Waittime" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Fixation* updates
            
            # if Fixation is starting this frame...
            if Fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Fixation.frameNStart = frameN  # exact frame index
                Fixation.tStart = t  # local t and not account for scr refresh
                Fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Fixation.started')
                # update status
                Fixation.status = STARTED
                Fixation.setAutoDraw(True)
            
            # if Fixation is active this frame...
            if Fixation.status == STARTED:
                # update params
                pass
            
            # if Fixation is stopping this frame...
            if Fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Fixation.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    Fixation.tStop = t  # not accounting for scr refresh
                    Fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Fixation.stopped')
                    # update status
                    Fixation.status = FINISHED
                    Fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in WaittimeComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Waittime" ---
        for thisComponent in WaittimeComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Waittime.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # set up handler to look after randomisation of conditions etc
        Runs = data.TrialHandler(nReps=1.0, method='fullRandom', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='Runs')
        thisExp.addLoop(Runs)  # add the loop to the experiment
        thisRun = Runs.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        for thisRun in Runs:
            currentLoop = Runs
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
            if thisRun != None:
                for paramName in thisRun:
                    globals()[paramName] = thisRun[paramName]
            
            # set up handler to look after randomisation of conditions etc
            VSSBlockRepeat1 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='VSSBlockRepeat1')
            thisExp.addLoop(VSSBlockRepeat1)  # add the loop to the experiment
            thisVSSBlockRepeat1 = VSSBlockRepeat1.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisVSSBlockRepeat1.rgb)
            if thisVSSBlockRepeat1 != None:
                for paramName in thisVSSBlockRepeat1:
                    globals()[paramName] = thisVSSBlockRepeat1[paramName]
            
            for thisVSSBlockRepeat1 in VSSBlockRepeat1:
                currentLoop = VSSBlockRepeat1
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisVSSBlockRepeat1.rgb)
                if thisVSSBlockRepeat1 != None:
                    for paramName in thisVSSBlockRepeat1:
                        globals()[paramName] = thisVSSBlockRepeat1[paramName]
                
                # set up handler to look after randomisation of conditions etc
                VSSBlock1 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/VSS_PsychoPy_Task_Parameters_Use_This_One.xlsx', selection=random(12)*503),
                    seed=None, name='VSSBlock1')
                thisExp.addLoop(VSSBlock1)  # add the loop to the experiment
                thisVSSBlock1 = VSSBlock1.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisVSSBlock1.rgb)
                if thisVSSBlock1 != None:
                    for paramName in thisVSSBlock1:
                        globals()[paramName] = thisVSSBlock1[paramName]
                
                for thisVSSBlock1 in VSSBlock1:
                    currentLoop = VSSBlock1
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisVSSBlock1.rgb)
                    if thisVSSBlock1 != None:
                        for paramName in thisVSSBlock1:
                            globals()[paramName] = thisVSSBlock1[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    VSSBlock1.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    VSSBlock1.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'VSSBlock1'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'VSSBlockRepeat1'
            
            
            # set up handler to look after randomisation of conditions etc
            VSSBlockRepeat2 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='VSSBlockRepeat2')
            thisExp.addLoop(VSSBlockRepeat2)  # add the loop to the experiment
            thisVSSBlockRepeat2 = VSSBlockRepeat2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisVSSBlockRepeat2.rgb)
            if thisVSSBlockRepeat2 != None:
                for paramName in thisVSSBlockRepeat2:
                    globals()[paramName] = thisVSSBlockRepeat2[paramName]
            
            for thisVSSBlockRepeat2 in VSSBlockRepeat2:
                currentLoop = VSSBlockRepeat2
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisVSSBlockRepeat2.rgb)
                if thisVSSBlockRepeat2 != None:
                    for paramName in thisVSSBlockRepeat2:
                        globals()[paramName] = thisVSSBlockRepeat2[paramName]
                
                # set up handler to look after randomisation of conditions etc
                VSSBlock2 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/VSS_PsychoPy_Task_Parameters_Use_This_One.xlsx', selection=random(12)*503),
                    seed=None, name='VSSBlock2')
                thisExp.addLoop(VSSBlock2)  # add the loop to the experiment
                thisVSSBlock2 = VSSBlock2.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisVSSBlock2.rgb)
                if thisVSSBlock2 != None:
                    for paramName in thisVSSBlock2:
                        globals()[paramName] = thisVSSBlock2[paramName]
                
                for thisVSSBlock2 in VSSBlock2:
                    currentLoop = VSSBlock2
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisVSSBlock2.rgb)
                    if thisVSSBlock2 != None:
                        for paramName in thisVSSBlock2:
                            globals()[paramName] = thisVSSBlock2[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    VSSBlock2.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    VSSBlock2.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'VSSBlock2'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'VSSBlockRepeat2'
            
            
            # set up handler to look after randomisation of conditions etc
            LetterBlockRepeat1 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='LetterBlockRepeat1')
            thisExp.addLoop(LetterBlockRepeat1)  # add the loop to the experiment
            thisLetterBlockRepeat1 = LetterBlockRepeat1.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisLetterBlockRepeat1.rgb)
            if thisLetterBlockRepeat1 != None:
                for paramName in thisLetterBlockRepeat1:
                    globals()[paramName] = thisLetterBlockRepeat1[paramName]
            
            for thisLetterBlockRepeat1 in LetterBlockRepeat1:
                currentLoop = LetterBlockRepeat1
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisLetterBlockRepeat1.rgb)
                if thisLetterBlockRepeat1 != None:
                    for paramName in thisLetterBlockRepeat1:
                        globals()[paramName] = thisLetterBlockRepeat1[paramName]
                
                # set up handler to look after randomisation of conditions etc
                LetterBlock1 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/LetterNaming.xlsx', selection=random(12)*503),
                    seed=None, name='LetterBlock1')
                thisExp.addLoop(LetterBlock1)  # add the loop to the experiment
                thisLetterBlock1 = LetterBlock1.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisLetterBlock1.rgb)
                if thisLetterBlock1 != None:
                    for paramName in thisLetterBlock1:
                        globals()[paramName] = thisLetterBlock1[paramName]
                
                for thisLetterBlock1 in LetterBlock1:
                    currentLoop = LetterBlock1
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisLetterBlock1.rgb)
                    if thisLetterBlock1 != None:
                        for paramName in thisLetterBlock1:
                            globals()[paramName] = thisLetterBlock1[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    LetterBlock1.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    LetterBlock1.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'LetterBlock1'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'LetterBlockRepeat1'
            
            
            # set up handler to look after randomisation of conditions etc
            LetterBlockRepeat2 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='LetterBlockRepeat2')
            thisExp.addLoop(LetterBlockRepeat2)  # add the loop to the experiment
            thisLetterBlockRepeat2 = LetterBlockRepeat2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisLetterBlockRepeat2.rgb)
            if thisLetterBlockRepeat2 != None:
                for paramName in thisLetterBlockRepeat2:
                    globals()[paramName] = thisLetterBlockRepeat2[paramName]
            
            for thisLetterBlockRepeat2 in LetterBlockRepeat2:
                currentLoop = LetterBlockRepeat2
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisLetterBlockRepeat2.rgb)
                if thisLetterBlockRepeat2 != None:
                    for paramName in thisLetterBlockRepeat2:
                        globals()[paramName] = thisLetterBlockRepeat2[paramName]
                
                # set up handler to look after randomisation of conditions etc
                LetterBlock2 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/LetterNaming.xlsx', selection=random(12)*503),
                    seed=None, name='LetterBlock2')
                thisExp.addLoop(LetterBlock2)  # add the loop to the experiment
                thisLetterBlock2 = LetterBlock2.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisLetterBlock2.rgb)
                if thisLetterBlock2 != None:
                    for paramName in thisLetterBlock2:
                        globals()[paramName] = thisLetterBlock2[paramName]
                
                for thisLetterBlock2 in LetterBlock2:
                    currentLoop = LetterBlock2
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisLetterBlock2.rgb)
                    if thisLetterBlock2 != None:
                        for paramName in thisLetterBlock2:
                            globals()[paramName] = thisLetterBlock2[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    LetterBlock2.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    LetterBlock2.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'LetterBlock2'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'LetterBlockRepeat2'
            
            
            # set up handler to look after randomisation of conditions etc
            NumberBlockRepeat1 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='NumberBlockRepeat1')
            thisExp.addLoop(NumberBlockRepeat1)  # add the loop to the experiment
            thisNumberBlockRepeat1 = NumberBlockRepeat1.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisNumberBlockRepeat1.rgb)
            if thisNumberBlockRepeat1 != None:
                for paramName in thisNumberBlockRepeat1:
                    globals()[paramName] = thisNumberBlockRepeat1[paramName]
            
            for thisNumberBlockRepeat1 in NumberBlockRepeat1:
                currentLoop = NumberBlockRepeat1
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisNumberBlockRepeat1.rgb)
                if thisNumberBlockRepeat1 != None:
                    for paramName in thisNumberBlockRepeat1:
                        globals()[paramName] = thisNumberBlockRepeat1[paramName]
                
                # set up handler to look after randomisation of conditions etc
                NumberBlock1 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/NumberNaming.xlsx', selection=random(12)*503),
                    seed=None, name='NumberBlock1')
                thisExp.addLoop(NumberBlock1)  # add the loop to the experiment
                thisNumberBlock1 = NumberBlock1.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisNumberBlock1.rgb)
                if thisNumberBlock1 != None:
                    for paramName in thisNumberBlock1:
                        globals()[paramName] = thisNumberBlock1[paramName]
                
                for thisNumberBlock1 in NumberBlock1:
                    currentLoop = NumberBlock1
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisNumberBlock1.rgb)
                    if thisNumberBlock1 != None:
                        for paramName in thisNumberBlock1:
                            globals()[paramName] = thisNumberBlock1[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    NumberBlock1.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    NumberBlock1.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'NumberBlock1'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'NumberBlockRepeat1'
            
            
            # set up handler to look after randomisation of conditions etc
            NumberBlockRepeat2 = data.TrialHandler(nReps=1.0, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='NumberBlockRepeat2')
            thisExp.addLoop(NumberBlockRepeat2)  # add the loop to the experiment
            thisNumberBlockRepeat2 = NumberBlockRepeat2.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisNumberBlockRepeat2.rgb)
            if thisNumberBlockRepeat2 != None:
                for paramName in thisNumberBlockRepeat2:
                    globals()[paramName] = thisNumberBlockRepeat2[paramName]
            
            for thisNumberBlockRepeat2 in NumberBlockRepeat2:
                currentLoop = NumberBlockRepeat2
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisNumberBlockRepeat2.rgb)
                if thisNumberBlockRepeat2 != None:
                    for paramName in thisNumberBlockRepeat2:
                        globals()[paramName] = thisNumberBlockRepeat2[paramName]
                
                # set up handler to look after randomisation of conditions etc
                NumberBlock2 = data.TrialHandler(nReps=1.0, method='sequential', 
                    extraInfo=expInfo, originPath=-1,
                    trialList=data.importConditions('Task Setup Files/NumberNaming.xlsx', selection=random(12)*503),
                    seed=None, name='NumberBlock2')
                thisExp.addLoop(NumberBlock2)  # add the loop to the experiment
                thisNumberBlock2 = NumberBlock2.trialList[0]  # so we can initialise stimuli with some values
                # abbreviate parameter names if possible (e.g. rgb = thisNumberBlock2.rgb)
                if thisNumberBlock2 != None:
                    for paramName in thisNumberBlock2:
                        globals()[paramName] = thisNumberBlock2[paramName]
                
                for thisNumberBlock2 in NumberBlock2:
                    currentLoop = NumberBlock2
                    thisExp.timestampOnFlip(win, 'thisRow.t')
                    # pause experiment here if requested
                    if thisExp.status == PAUSED:
                        pauseExperiment(
                            thisExp=thisExp, 
                            inputs=inputs, 
                            win=win, 
                            timers=[routineTimer], 
                            playbackComponents=[]
                    )
                    # abbreviate parameter names if possible (e.g. rgb = thisNumberBlock2.rgb)
                    if thisNumberBlock2 != None:
                        for paramName in thisNumberBlock2:
                            globals()[paramName] = thisNumberBlock2[paramName]
                    
                    # --- Prepare to start Routine "trial" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('trial.started', globalClock.getTime())
                    vsstask.reset()
                    vsstask.setFillColor(Background)
                    vsstask.setText(Combined)
                    # keep track of which components have finished
                    trialComponents = [vsstask, mic]
                    for thisComponent in trialComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "trial" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 2.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *vsstask* updates
                        
                        # if vsstask is starting this frame...
                        if vsstask.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            vsstask.frameNStart = frameN  # exact frame index
                            vsstask.tStart = t  # local t and not account for scr refresh
                            vsstask.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(vsstask, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'vsstask.started')
                            # update status
                            vsstask.status = STARTED
                            vsstask.setAutoDraw(True)
                        
                        # if vsstask is active this frame...
                        if vsstask.status == STARTED:
                            # update params
                            pass
                        
                        # if vsstask is stopping this frame...
                        if vsstask.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > vsstask.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                vsstask.tStop = t  # not accounting for scr refresh
                                vsstask.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'vsstask.stopped')
                                # update status
                                vsstask.status = FINISHED
                                vsstask.setAutoDraw(False)
                        
                        # if mic is starting this frame...
                        if mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mic.frameNStart = frameN  # exact frame index
                            mic.tStart = t  # local t and not account for scr refresh
                            mic.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mic, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.addData('mic.started', t)
                            # update status
                            mic.status = STARTED
                            # start recording with mic
                            mic.start()
                        
                        # if mic is active this frame...
                        if mic.status == STARTED:
                            # update params
                            pass
                            # update recorded clip for mic
                            mic.poll()
                        
                        # if mic is stopping this frame...
                        if mic.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > mic.tStartRefresh + 2-frameTolerance:
                                # keep track of stop time/frame for later
                                mic.tStop = t  # not accounting for scr refresh
                                mic.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.addData('mic.stopped', t)
                                # update status
                                mic.status = FINISHED
                                # stop recording with mic
                                mic.stop()
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in trialComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "trial" ---
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('trial.stopped', globalClock.getTime())
                    # tell mic to keep hold of current recording in mic.clips and transcript (if applicable) in mic.scripts
                    # this will also update mic.lastClip and mic.lastScript
                    mic.stop()
                    tag = data.utils.getDateStr()
                    micClip, micScript = mic.bank(
                        tag=tag, transcribe='whisper',
                        language='en-US', expectedWords=None
                    )
                    NumberBlock2.addData('mic.clip', os.path.join(micRecFolder, 'recording_mic_%s.wav' % tag))
                    NumberBlock2.addData('mic.script', micScript)
                    # save transcription data
                    with open(os.path.join(micRecFolder, 'recording_mic_%s.json' % tag), 'w') as fp:
                        fp.write(micScript.response)
                    # save speaking start/stop times
                    micWordData = []
                    micSegments = mic.lastScript.responseData.get('segments', {})
                    for thisSegment in micSegments.values():
                        # for each segment...
                        for thisWord in thisSegment.get('words', {}).values():
                            # append word data
                            micWordData.append(thisWord)
                    # if there were any words, store the start of first & end of last 
                    if len(micWordData):
                        thisExp.addData('mic.speechStart', micWordData[0]['start'])
                        thisExp.addData('mic.speechEnd', micWordData[-1]['end'])
                    else:
                        thisExp.addData('mic.speechStart', '')
                        thisExp.addData('mic.speechEnd', '')
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-2.000000)
                    
                    # --- Prepare to start Routine "halfsecfix" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('halfsecfix.started', globalClock.getTime())
                    text_2.setText('+')
                    # keep track of which components have finished
                    halfsecfixComponents = [text_2]
                    for thisComponent in halfsecfixComponents:
                        thisComponent.tStart = None
                        thisComponent.tStop = None
                        thisComponent.tStartRefresh = None
                        thisComponent.tStopRefresh = None
                        if hasattr(thisComponent, 'status'):
                            thisComponent.status = NOT_STARTED
                    # reset timers
                    t = 0
                    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                    frameN = -1
                    
                    # --- Run Routine "halfsecfix" ---
                    routineForceEnded = not continueRoutine
                    while continueRoutine and routineTimer.getTime() < 0.5:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *text_2* updates
                        
                        # if text_2 is starting this frame...
                        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_2.frameNStart = frameN  # exact frame index
                            text_2.tStart = t  # local t and not account for scr refresh
                            text_2.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text_2.started')
                            # update status
                            text_2.status = STARTED
                            text_2.setAutoDraw(True)
                        
                        # if text_2 is active this frame...
                        if text_2.status == STARTED:
                            # update params
                            pass
                        
                        # if text_2 is stopping this frame...
                        if text_2.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_2.tStartRefresh + 0.5-frameTolerance:
                                # keep track of stop time/frame for later
                                text_2.tStop = t  # not accounting for scr refresh
                                text_2.frameNStop = frameN  # exact frame index
                                # add timestamp to datafile
                                thisExp.timestampOnFlip(win, 'text_2.stopped')
                                # update status
                                text_2.status = FINISHED
                                text_2.setAutoDraw(False)
                        
                        # check for quit (typically the Esc key)
                        if defaultKeyboard.getKeys(keyList=["escape"]):
                            thisExp.status = FINISHED
                        if thisExp.status == FINISHED or endExpNow:
                            endExperiment(thisExp, inputs=inputs, win=win)
                            return
                        
                        # check if all components have finished
                        if not continueRoutine:  # a component has requested a forced-end of Routine
                            routineForceEnded = True
                            break
                        continueRoutine = False  # will revert to True if at least one component still running
                        for thisComponent in halfsecfixComponents:
                            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                                continueRoutine = True
                                break  # at least one component has not yet finished
                        
                        # refresh the screen
                        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                            win.flip()
                    
                    # --- Ending Routine "halfsecfix" ---
                    for thisComponent in halfsecfixComponents:
                        if hasattr(thisComponent, "setAutoDraw"):
                            thisComponent.setAutoDraw(False)
                    thisExp.addData('halfsecfix.stopped', globalClock.getTime())
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-0.500000)
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed 1.0 repeats of 'NumberBlock2'
                
                
                # --- Prepare to start Routine "BlockRest" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('BlockRest.started', globalClock.getTime())
                # keep track of which components have finished
                BlockRestComponents = [text]
                for thisComponent in BlockRestComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "BlockRest" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 20.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text* updates
                    
                    # if text is starting this frame...
                    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text.frameNStart = frameN  # exact frame index
                        text.tStart = t  # local t and not account for scr refresh
                        text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.started')
                        # update status
                        text.status = STARTED
                        text.setAutoDraw(True)
                    
                    # if text is active this frame...
                    if text.status == STARTED:
                        # update params
                        pass
                    
                    # if text is stopping this frame...
                    if text.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text.tStartRefresh + 20-frameTolerance:
                            # keep track of stop time/frame for later
                            text.tStop = t  # not accounting for scr refresh
                            text.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'text.stopped')
                            # update status
                            text.status = FINISHED
                            text.setAutoDraw(False)
                    
                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in BlockRestComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "BlockRest" ---
                for thisComponent in BlockRestComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('BlockRest.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-20.000000)
            # completed 1.0 repeats of 'NumberBlockRepeat2'
            
        # completed 1.0 repeats of 'Runs'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 3.0 repeats of 'Run_repeat'
    
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    # save mic recordings
    for tag in mic.clips:
        for i, clip in enumerate(mic.clips[tag]):
            clipFilename = 'recording_mic_%s.wav' % tag
            # if there's more than 1 clip with this tag, append a counter for all beyond the first
            if i > 0:
                clipFilename += '_%s' % i
            clip.save(os.path.join(micRecFolder, clipFilename))
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
