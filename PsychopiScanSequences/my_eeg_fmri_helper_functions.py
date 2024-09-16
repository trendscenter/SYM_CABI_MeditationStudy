
# interfacing with BrainProducts EEG
import time
from psychopy.hardware import brainproducts

# interfacing with triggerbox
from psychopy import core
from psychopy.hardware import keyboard
from psychopy.visual import Window, Rect
from pylsl import StreamInfo, StreamOutlet
import serial


CONNECT_TO_RCS_SERVER = False
print(f'CONNECT_TO_RCS_SERVER : {CONNECT_TO_RCS_SERVER}')

if not CONNECT_TO_RCS_SERVER:
    print("You are not connected to RCS Server. This run with just print outputs.. ")
    #var = input("You are not connected to RCS Server. This run with just print outputs.. Do you want to continue [y|Y] : ")
    #if var.lower() != 'y':
    #    quit()


def task_trigger(value, port, outlet, desc=""):
    if  CONNECT_TO_RCS_SERVER:
        port.write(value);outlet.push_sample(value);
    else:
        print(value)

def eeg_setup(expName, participant):
    if CONNECT_TO_RCS_SERVER:
        port = serial.Serial("COM12")
        info = StreamInfo(name="LSL_Markers", type="Markers", channel_count=1,
                          channel_format="int32", source_id="LSL_Markers_001")
        outlet = StreamOutlet(info)

        workspace = 'C:/Users/CABI Users/Desktop/user_experiments/calhoun/crest_tms/workspace_files/BrainCapMR_64channel_workspace.rwksp';
        workspace = 'C:/Users/CABI Users/Desktop/user_experiments/calhoun/crest_tms/workspace_files/net32.rwksp';

        print('Participant is: ', participant);
        rcs = brainproducts.RemoteControlServer(host='143.215.238.204', timeout=20)
        rcs.open(expName, workspace=workspace, participant=participant)

        # Open BrainVision Recorder with the previously entered settings
        rcs.openRecorder()
        time.sleep(2)

        rcs.mode = 'monitor'
        print(rcs.recordingState)
        time.sleep(2)

        # Start test mode (sine waves and occasional triggers)
        rcs.mode = 'test'
        print(rcs.recordingState)
        time.sleep(2)

        # Start impedance check mode
        rcs.mode = 'impedance'
        print(rcs.recordingState)
        time.sleep(10)

        # Start recording
        rcs.startRecording()
        print(rcs.recordingState)
        time.sleep(6)
        rcs_recording_flag=True

        return port, outlet, rcs, rcs_recording_flag

    else:
        print(f'Not connected to server.. Called eeg_setup().. when connected to server, connects to RCS server and sets values for "PORT", "OUTLET", "RCS" and "rcs_recording_flag"')
        return "PORT", "OUTLET", "RCS", "rcs_recording_flag"


def eeg_at_close(port, outlet, rcs, rcs_recording_flag):
    if CONNECT_TO_RCS_SERVER:
        if rcs_recording_flag:
            rcs.stopRecording()
            time.sleep(1)

        # Go back to idle state
        rcs.mode = 'default';
        time.sleep(2);
        rcs.close()

        port.close()
    else:
        print(f'Not connected to server.. Called eeg_at_close().. when connected to RCS server closes {port}, {rcs}, {outlet} and {rcs_recording_flag}')



if __name__ == '__main__':
    print("..Testing utils..")
