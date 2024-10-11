#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Fri Oct 11 13:24:26 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019)
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195.
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins

plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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

# Run 'Before Experiment' code from code_2
import numpy as np
import random
import my_eeg_fmri_helper_functions as hf

high_imgs_order = np.random.permutation(18).reshape((6, 3)).tolist()

low_imgs_order = np.random.permutation(18).reshape((6, 3)).tolist()

blocks = ['H'] * 6 + ['L'] * 6
# Code to make sure all Low valence or high valence do not appear together
while (''.join(blocks) == 'HHHHHHLLLLLL' or ''.join(blocks) == 'LLLLLLHHHHHH'):
    random.shuffle(blocks)
print(''.join(blocks))

global high_nReps
global low_nReps
high_nReps = 1
low_nReps = 0

if blocks[0] == 'L':
    high_nReps = 0
    low_nReps = 1

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'CABI_SYM_AversiveVisualStimuliTask'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.

    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
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
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)

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
        originPath='./data/4_AversiveVisualStimuliTask_lastrun.py',
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
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename + '.log', level=_loggingLevel)

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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')

    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1, -1, -1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'

    win._monitorFrameRate = hf.PSYCHOPY_EXP_DEFAULT_FRAMERATE
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    print(f'Current frameRate: {expInfo["frameRate"]}')

    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()

    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to
    the device manager (deviceManager)

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
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}

    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')

    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer

    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True


def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.

    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
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
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
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
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
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

    # --- Initialize components for Routine "dummy_start_buffer_1" ---
    text = visual.TextStim(win=win, name='text',
                           text=None,
                           font='Arial',
                           pos=(0, 0), height=0.8, wrapWidth=None, ori=0.0,
                           color='white', colorSpace='rgb', opacity=None,
                           languageStyle='LTR',
                           depth=0.0);

    # --- Initialize components for Routine "dummy_test" ---
    # Run 'Begin Experiment' code from code_2
    global high_nReps
    global low_nReps

    # --- Initialize components for Routine "high_val_imgs" ---
    high_val_img = visual.ImageStim(
        win=win,
        name='high_val_img',
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "high_blank" ---
    high_blank_img = visual.ImageStim(
        win=win,
        name='high_blank_img',
        image='images/blank_screen.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "high_fixation_cross" ---
    high_fixation_img = visual.ImageStim(
        win=win,
        name='high_fixation_img',
        image='images/fixation_cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "low_val_imgs" ---
    low_val_img = visual.ImageStim(
        win=win,
        name='low_val_img',
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "low_blank" ---
    low_blank_image = visual.ImageStim(
        win=win,
        name='low_blank_image',
        image='images/blank_screen.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "low_fixation_cross" ---
    low_fixation_img = visual.ImageStim(
        win=win,
        name='low_fixation_img',
        image='images/fixation_cross.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1, 1, 1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)

    # --- Initialize components for Routine "dummy_end" ---
    text_2 = visual.TextStim(win=win, name='text_2',
                             text=None,
                             font='Arial',
                             pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0,
                             color='white', colorSpace='rgb', opacity=None,
                             languageStyle='LTR',
                             depth=0.0);

    # create some handy timers

    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )

    # --- Prepare to start Routine "dummy_start_buffer_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('dummy_start_buffer_1.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    dummy_start_buffer_1Components = [text]
    for thisComponent in dummy_start_buffer_1Components:
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

    # --- Run Routine "dummy_start_buffer_1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *text* updates

        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
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
            if tThisFlipGlobal > text.tStartRefresh + 10 - frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        # Run 'Each Frame' code from code_6
        text.text = 10 - int(t)

        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in dummy_start_buffer_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "dummy_start_buffer_1" ---
    for thisComponent in dummy_start_buffer_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('dummy_start_buffer_1.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    thisExp.nextEntry()

    # set up handler to look after randomisation of conditions etc
    block_repeat = data.TrialHandler(nReps=12.0, method='sequential',
                                     extraInfo=expInfo, originPath=-1,
                                     trialList=[None],
                                     seed=None, name='block_repeat')
    thisExp.addLoop(block_repeat)  # add the loop to the experiment
    thisBlock_repeat = block_repeat.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_repeat.rgb)
    if thisBlock_repeat != None:
        for paramName in thisBlock_repeat:
            globals()[paramName] = thisBlock_repeat[paramName]

    for thisBlock_repeat in block_repeat:
        currentLoop = block_repeat
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp,
                win=win,
                timers=[routineTimer],
                playbackComponents=[]
            )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_repeat.rgb)
        if thisBlock_repeat != None:
            for paramName in thisBlock_repeat:
                globals()[paramName] = thisBlock_repeat[paramName]

        # --- Prepare to start Routine "dummy_test" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('dummy_test.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from code_2
        if blocks[block_repeat.thisN] == 'H':
            high_nReps = 1
            low_nReps = 0
        elif blocks[block_repeat.thisN] == 'L':
            high_nReps = 0
            low_nReps = 1
        else:
            raise Exception('Invalid block ID. It should be either "H" or "L"')

        # keep track of which components have finished
        dummy_testComponents = []
        for thisComponent in dummy_testComponents:
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

        # --- Run Routine "dummy_test" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame

            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return

            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in dummy_testComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished

            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()

        # --- Ending Routine "dummy_test" ---
        for thisComponent in dummy_testComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('dummy_test.stopped', globalClock.getTime(format='float'))
        # the Routine "dummy_test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()

        # set up handler to look after randomisation of conditions etc
        high_block = data.TrialHandler(nReps=high_nReps, method='sequential',
                                       extraInfo=expInfo, originPath=-1,
                                       trialList=[None],
                                       seed=None, name='high_block')
        thisExp.addLoop(high_block)  # add the loop to the experiment
        thisHigh_block = high_block.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisHigh_block.rgb)
        if thisHigh_block != None:
            for paramName in thisHigh_block:
                globals()[paramName] = thisHigh_block[paramName]

        for thisHigh_block in high_block:
            currentLoop = high_block
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[]
                )
            # abbreviate parameter names if possible (e.g. rgb = thisHigh_block.rgb)
            if thisHigh_block != None:
                for paramName in thisHigh_block:
                    globals()[paramName] = thisHigh_block[paramName]

            # set up handler to look after randomisation of conditions etc
            within_high_block = data.TrialHandler(nReps=1.0, method='sequential',
                                                  extraInfo=expInfo, originPath=-1,
                                                  trialList=data.importConditions('high_valence.xlsx',
                                                                                  selection=high_imgs_order.pop()),
                                                  seed=None, name='within_high_block')
            thisExp.addLoop(within_high_block)  # add the loop to the experiment
            thisWithin_high_block = within_high_block.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisWithin_high_block.rgb)
            if thisWithin_high_block != None:
                for paramName in thisWithin_high_block:
                    globals()[paramName] = thisWithin_high_block[paramName]

            for thisWithin_high_block in within_high_block:
                currentLoop = within_high_block
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[]
                    )
                # abbreviate parameter names if possible (e.g. rgb = thisWithin_high_block.rgb)
                if thisWithin_high_block != None:
                    for paramName in thisWithin_high_block:
                        globals()[paramName] = thisWithin_high_block[paramName]

                # --- Prepare to start Routine "high_val_imgs" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('high_val_imgs.started', globalClock.getTime(format='float'))
                high_val_img.setImage(image_file)
                # Run 'Begin Routine' code from code
                ### START ROUTINE SBASODI1 (HIGH)
                hf.task_trigger(value="3", port=port, outlet=outlet, desc="High Valence Image")
                ### END OF START ROUTINE CODE SBASODI1 (HIGH)
                # keep track of which components have finished
                high_val_imgsComponents = [high_val_img]
                for thisComponent in high_val_imgsComponents:
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

                # --- Run Routine "high_val_imgs" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 6.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame

                    # *high_val_img* updates

                    # if high_val_img is starting this frame...
                    if high_val_img.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                        # keep track of start time/frame for later
                        high_val_img.frameNStart = frameN  # exact frame index
                        high_val_img.tStart = t  # local t and not account for scr refresh
                        high_val_img.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(high_val_img, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'high_val_img.started')
                        # update status
                        high_val_img.status = STARTED
                        high_val_img.setAutoDraw(True)

                    # if high_val_img is active this frame...
                    if high_val_img.status == STARTED:
                        # update params
                        pass

                    # if high_val_img is stopping this frame...
                    if high_val_img.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > high_val_img.tStartRefresh + 6 - frameTolerance:
                            # keep track of stop time/frame for later
                            high_val_img.tStop = t  # not accounting for scr refresh
                            high_val_img.tStopRefresh = tThisFlipGlobal  # on global time
                            high_val_img.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'high_val_img.stopped')
                            # update status
                            high_val_img.status = FINISHED
                            high_val_img.setAutoDraw(False)

                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in high_val_imgsComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished

                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # --- Ending Routine "high_val_imgs" ---
                for thisComponent in high_val_imgsComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('high_val_imgs.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from code
                ### END ROUTINE SBASODI1 (HIGH)
                hf.task_trigger(value="4", port=port, outlet=outlet, desc="High Valence Image")
                ### END OF END ROUTINE CODE SBASODI1 (HIGH)

                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-6.000000)

                # --- Prepare to start Routine "high_blank" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('high_blank.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                high_blankComponents = [high_blank_img]
                for thisComponent in high_blankComponents:
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

                # --- Run Routine "high_blank" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 3.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame

                    # *high_blank_img* updates

                    # if high_blank_img is starting this frame...
                    if high_blank_img.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                        # keep track of start time/frame for later
                        high_blank_img.frameNStart = frameN  # exact frame index
                        high_blank_img.tStart = t  # local t and not account for scr refresh
                        high_blank_img.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(high_blank_img, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'high_blank_img.started')
                        # update status
                        high_blank_img.status = STARTED
                        high_blank_img.setAutoDraw(True)

                    # if high_blank_img is active this frame...
                    if high_blank_img.status == STARTED:
                        # update params
                        pass

                    # if high_blank_img is stopping this frame...
                    if high_blank_img.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > high_blank_img.tStartRefresh + 3.0 - frameTolerance:
                            # keep track of stop time/frame for later
                            high_blank_img.tStop = t  # not accounting for scr refresh
                            high_blank_img.tStopRefresh = tThisFlipGlobal  # on global time
                            high_blank_img.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'high_blank_img.stopped')
                            # update status
                            high_blank_img.status = FINISHED
                            high_blank_img.setAutoDraw(False)

                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in high_blankComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished

                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # --- Ending Routine "high_blank" ---
                for thisComponent in high_blankComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('high_blank.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-3.000000)
                thisExp.nextEntry()

                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1.0 repeats of 'within_high_block'

            # --- Prepare to start Routine "high_fixation_cross" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('high_fixation_cross.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_4
            ### START ROUTINE SBASODI1 (HIGH_FIXATION)
            hf.task_trigger(value="5", port=port, outlet=outlet, desc="High Fixation Image")

            ### START OF END ROUTINE CODE SBASODI1 (HIGH_FIXATION)

            # keep track of which components have finished
            high_fixation_crossComponents = [high_fixation_img]
            for thisComponent in high_fixation_crossComponents:
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

            # --- Run Routine "high_fixation_cross" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 13.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *high_fixation_img* updates

                # if high_fixation_img is starting this frame...
                if high_fixation_img.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    high_fixation_img.frameNStart = frameN  # exact frame index
                    high_fixation_img.tStart = t  # local t and not account for scr refresh
                    high_fixation_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(high_fixation_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'high_fixation_img.started')
                    # update status
                    high_fixation_img.status = STARTED
                    high_fixation_img.setAutoDraw(True)

                # if high_fixation_img is active this frame...
                if high_fixation_img.status == STARTED:
                    # update params
                    pass

                # if high_fixation_img is stopping this frame...
                if high_fixation_img.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > high_fixation_img.tStartRefresh + 13 - frameTolerance:
                        # keep track of stop time/frame for later
                        high_fixation_img.tStop = t  # not accounting for scr refresh
                        high_fixation_img.tStopRefresh = tThisFlipGlobal  # on global time
                        high_fixation_img.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'high_fixation_img.stopped')
                        # update status
                        high_fixation_img.status = FINISHED
                        high_fixation_img.setAutoDraw(False)

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in high_fixation_crossComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "high_fixation_cross" ---
            for thisComponent in high_fixation_crossComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('high_fixation_cross.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_4
            ### END ROUTINE SBASODI1 (HIGH_FIXATION)
            hf.task_trigger(value="6", port=port, outlet=outlet, desc="High Fixation Image")

            ### END OF END ROUTINE CODE SBASODI1 (HIGH_FIXATION)

            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-13.000000)
            thisExp.nextEntry()

            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed high_nReps repeats of 'high_block'

        # set up handler to look after randomisation of conditions etc
        low_block = data.TrialHandler(nReps=low_nReps, method='sequential',
                                      extraInfo=expInfo, originPath=-1,
                                      trialList=[None],
                                      seed=None, name='low_block')
        thisExp.addLoop(low_block)  # add the loop to the experiment
        thisLow_block = low_block.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLow_block.rgb)
        if thisLow_block != None:
            for paramName in thisLow_block:
                globals()[paramName] = thisLow_block[paramName]

        for thisLow_block in low_block:
            currentLoop = low_block
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp,
                    win=win,
                    timers=[routineTimer],
                    playbackComponents=[]
                )
            # abbreviate parameter names if possible (e.g. rgb = thisLow_block.rgb)
            if thisLow_block != None:
                for paramName in thisLow_block:
                    globals()[paramName] = thisLow_block[paramName]

            # set up handler to look after randomisation of conditions etc
            within_low_block = data.TrialHandler(nReps=1.0, method='sequential',
                                                 extraInfo=expInfo, originPath=-1,
                                                 trialList=data.importConditions('low_valence.xlsx',
                                                                                 selection=low_imgs_order.pop()),
                                                 seed=None, name='within_low_block')
            thisExp.addLoop(within_low_block)  # add the loop to the experiment
            thisWithin_low_block = within_low_block.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisWithin_low_block.rgb)
            if thisWithin_low_block != None:
                for paramName in thisWithin_low_block:
                    globals()[paramName] = thisWithin_low_block[paramName]

            for thisWithin_low_block in within_low_block:
                currentLoop = within_low_block
                thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp,
                        win=win,
                        timers=[routineTimer],
                        playbackComponents=[]
                    )
                # abbreviate parameter names if possible (e.g. rgb = thisWithin_low_block.rgb)
                if thisWithin_low_block != None:
                    for paramName in thisWithin_low_block:
                        globals()[paramName] = thisWithin_low_block[paramName]

                # --- Prepare to start Routine "low_val_imgs" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('low_val_imgs.started', globalClock.getTime(format='float'))
                low_val_img.setImage(image_file)
                # Run 'Begin Routine' code from code_3
                ### START ROUTINE SBASODI1 (LOW)
                hf.task_trigger(value="3", port=port, outlet=outlet, desc="Low Valence Image")

                ### END OF START ROUTINE CODE SBASODI1 (LOW)
                # keep track of which components have finished
                low_val_imgsComponents = [low_val_img]
                for thisComponent in low_val_imgsComponents:
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

                # --- Run Routine "low_val_imgs" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 6.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame

                    # *low_val_img* updates

                    # if low_val_img is starting this frame...
                    if low_val_img.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                        # keep track of start time/frame for later
                        low_val_img.frameNStart = frameN  # exact frame index
                        low_val_img.tStart = t  # local t and not account for scr refresh
                        low_val_img.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(low_val_img, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'low_val_img.started')
                        # update status
                        low_val_img.status = STARTED
                        low_val_img.setAutoDraw(True)

                    # if low_val_img is active this frame...
                    if low_val_img.status == STARTED:
                        # update params
                        pass

                    # if low_val_img is stopping this frame...
                    if low_val_img.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > low_val_img.tStartRefresh + 6 - frameTolerance:
                            # keep track of stop time/frame for later
                            low_val_img.tStop = t  # not accounting for scr refresh
                            low_val_img.tStopRefresh = tThisFlipGlobal  # on global time
                            low_val_img.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'low_val_img.stopped')
                            # update status
                            low_val_img.status = FINISHED
                            low_val_img.setAutoDraw(False)

                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in low_val_imgsComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished

                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # --- Ending Routine "low_val_imgs" ---
                for thisComponent in low_val_imgsComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('low_val_imgs.stopped', globalClock.getTime(format='float'))
                # Run 'End Routine' code from code_3
                ### END ROUTINE SBASODI1 (LOW)
                hf.task_trigger(value="4", port=port, outlet=outlet, desc="Low Valence Image")

                ### END OF END ROUTINE CODE SBASODI1 (LOW)

                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-6.000000)

                # --- Prepare to start Routine "low_blank" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('low_blank.started', globalClock.getTime(format='float'))
                # keep track of which components have finished
                low_blankComponents = [low_blank_image]
                for thisComponent in low_blankComponents:
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

                # --- Run Routine "low_blank" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 3.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame

                    # *low_blank_image* updates

                    # if low_blank_image is starting this frame...
                    if low_blank_image.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                        # keep track of start time/frame for later
                        low_blank_image.frameNStart = frameN  # exact frame index
                        low_blank_image.tStart = t  # local t and not account for scr refresh
                        low_blank_image.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(low_blank_image, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'low_blank_image.started')
                        # update status
                        low_blank_image.status = STARTED
                        low_blank_image.setAutoDraw(True)

                    # if low_blank_image is active this frame...
                    if low_blank_image.status == STARTED:
                        # update params
                        pass

                    # if low_blank_image is stopping this frame...
                    if low_blank_image.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > low_blank_image.tStartRefresh + 3.0 - frameTolerance:
                            # keep track of stop time/frame for later
                            low_blank_image.tStop = t  # not accounting for scr refresh
                            low_blank_image.tStopRefresh = tThisFlipGlobal  # on global time
                            low_blank_image.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'low_blank_image.stopped')
                            # update status
                            low_blank_image.status = FINISHED
                            low_blank_image.setAutoDraw(False)

                    # check for quit (typically the Esc key)
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, win=win)
                        return

                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in low_blankComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished

                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()

                # --- Ending Routine "low_blank" ---
                for thisComponent in low_blankComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('low_blank.stopped', globalClock.getTime(format='float'))
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-3.000000)
                thisExp.nextEntry()

                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1.0 repeats of 'within_low_block'

            # --- Prepare to start Routine "low_fixation_cross" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('low_fixation_cross.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from code_5
            ### START ROUTINE SBASODI1 (LOW_FIXATION)
            hf.task_trigger(value="5", port=port, outlet=outlet, desc="Low Fixation Image")

            ### START OF END ROUTINE CODE SBASODI1 (LOW_FIXATION)

            # keep track of which components have finished
            low_fixation_crossComponents = [low_fixation_img]
            for thisComponent in low_fixation_crossComponents:
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

            # --- Run Routine "low_fixation_cross" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 13.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *low_fixation_img* updates

                # if low_fixation_img is starting this frame...
                if low_fixation_img.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
                    # keep track of start time/frame for later
                    low_fixation_img.frameNStart = frameN  # exact frame index
                    low_fixation_img.tStart = t  # local t and not account for scr refresh
                    low_fixation_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(low_fixation_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'low_fixation_img.started')
                    # update status
                    low_fixation_img.status = STARTED
                    low_fixation_img.setAutoDraw(True)

                # if low_fixation_img is active this frame...
                if low_fixation_img.status == STARTED:
                    # update params
                    pass

                # if low_fixation_img is stopping this frame...
                if low_fixation_img.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > low_fixation_img.tStartRefresh + 13 - frameTolerance:
                        # keep track of stop time/frame for later
                        low_fixation_img.tStop = t  # not accounting for scr refresh
                        low_fixation_img.tStopRefresh = tThisFlipGlobal  # on global time
                        low_fixation_img.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'low_fixation_img.stopped')
                        # update status
                        low_fixation_img.status = FINISHED
                        low_fixation_img.setAutoDraw(False)

                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return

                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in low_fixation_crossComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished

                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()

            # --- Ending Routine "low_fixation_cross" ---
            for thisComponent in low_fixation_crossComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('low_fixation_cross.stopped', globalClock.getTime(format='float'))
            # Run 'End Routine' code from code_5
            ### END ROUTINE SBASODI1 (LOW_FIXATION)
            hf.task_trigger(value="6", port=port, outlet=outlet, desc="Low Fixation Image")

            ### END OF END ROUTINE CODE SBASODI1 (LOW_FIXATION)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-13.000000)
            thisExp.nextEntry()

            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed low_nReps repeats of 'low_block'

        thisExp.nextEntry()

        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 12.0 repeats of 'block_repeat'

    # --- Prepare to start Routine "dummy_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('dummy_end.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    dummy_endComponents = [text_2]
    for thisComponent in dummy_endComponents:
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

    # --- Run Routine "dummy_end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 120.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *text_2* updates

        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
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
            if tThisFlipGlobal > text_2.tStartRefresh + 120 - frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
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
            endExperiment(thisExp, win=win)
            return

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in dummy_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "dummy_end" ---
    for thisComponent in dummy_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('dummy_end.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-120.000000)
    thisExp.nextEntry()

    # mark experiment as finished
    endExperiment(thisExp, win=win)


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


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.

    This function does NOT close the window or end the Python process - use `quit` for this.

    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about
        where to save it to.
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
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.

    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
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
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
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
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)

    port, outlet, rcs, rcs_recording_flag = hf.eeg_setup(expName, expInfo.get('participant'))

    run(
        expInfo=expInfo,
        thisExp=thisExp,
        win=win,
        globalClock='float'
    )

    hf.eeg_at_close(port, outlet, rcs, rcs_recording_flag)

    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
