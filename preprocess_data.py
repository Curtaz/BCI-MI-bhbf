import glob
import os
import mne
import numpy as np
from neurorobotics_dl.events import Event
from neurorobotics_dl.utils import fix_mat
from scipy.io import loadmat,savemat
from scipy.signal import butter, lfilter

ALLOWED_FILE_FORMATS = ['mat','gdf']

"""
DATA_PATH = "/home/palatella/workspace/Neurorobotics-MI-data/d6"

OUT_PATH = os.path.join(DATA_PATH,"preprocessed")
LAP_PATH = "/home/palatella/workspace/BCI-MI-bhbf/config/laplacian16.mat"
# LAP_PATH = r"C:\Users\tomma\Documents\Uni\PhD\code\BCI-MI-bhbf\config\lapmask_antneuro_32.mat"
"""

DATA_PATH = r"C:\Users\tomma\Documents\Uni\PhD\data\d6"

OUT_PATH = os.path.join(DATA_PATH,"preprocessed")
LAP_PATH = r"C:\Users\tomma\Documents\Uni\PhD\code\BCI-MI-bhbf\config\laplacian16.mat"
# LAP_PATH = r"C:\Users\tomma\Documents\Uni\PhD\code\BCI-MI-bhbf\config\lapmask_antneuro_32.mat"
FILT_ORDER = 2
FC_H = 2
FC_L = 40
NUM_CHANNELS = 16
T_SHIFT = 0.0625
T_SIZE = 1

FILE_FORMATS = ["gdf","mat"]


def grab_files(root_dir,file_types=ALLOWED_FILE_FORMATS):
    files_grabbed = []
    for ftype in file_types:
        files_grabbed.extend(glob.glob(os.path.join(root_dir,f"*.{ftype}")))
    return files_grabbed

def main():
    for f in FILE_FORMATS:
        if f not in ALLOWED_FILE_FORMATS:
            raise Exception(f"Invalid file format {(f)} provided. Supported formats are", ','.join(ALLOWED_FILE_FORMATS))

    filenames = {"train": grab_files(os.path.join(DATA_PATH,"train")),
                "val": grab_files(os.path.join(DATA_PATH,"val")),
                "test": grab_files(os.path.join(DATA_PATH,"test"))}
    
    print()
    for k,v in filenames.items():
        print(f'{k}: found {len(v)} files')
    print()

    lap = loadmat(LAP_PATH)['lapmask']
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in ["train","val","test"]:
        if len(filenames[split])>0:
            print(f"{split}: reading from {os.path.join(DATA_PATH,split)}")
            eegs = [] 
            session_delims = []
            event_pos = []
            event_type = []
            last_eeg_len = 0
            dim_tot = 0
            i = 0

            for file in filenames[split]:
                print(file)
                file_format = file.split('.')[-1]
                if file_format == "gdf":
                    t_eeg,t_header = read_gdf(file)
                elif file_format == "mat":
                    t_eeg,t_header = read_mat(file)

                if check_sanity(t_eeg, t_header):
                    # Read data, setup configuration
                    events = t_header['EVENT']
                    fs = t_header['SampleRate']

                    # Preprocessing steps
                    # Apply Laplacian spatial filter
                    t_eeg = np.dot(t_eeg[:, :NUM_CHANNELS], lap)

                    # Apply bandpass filter
                    for ch in range(NUM_CHANNELS):
                        b, a = butter(FILT_ORDER, 2 * FC_L / fs, "low")
                        t_eeg[:, ch] = lfilter(b, a, t_eeg[:, ch])

                        d, c = butter(FILT_ORDER, 2 * FC_H / fs, "high")
                        t_eeg[:, ch] = lfilter(d, c, t_eeg[:, ch])
    
                    eegs.append(t_eeg)
                    event_pos.extend(events['POS'] + dim_tot)
                    event_type.extend(events['TYP'])
                    session_delims.append(last_eeg_len + len(t_eeg))
                    dim_tot += len(t_eeg)
                    i += 1
                else:
                    print("Bad format, corrupt file or whatever, skipping..")

            if dim_tot > 0:
                # EEG = np.full((dim_tot, NUM_CHANNELS), np.nan)
                # pos = 0
                # for i in range(len(eegs)):
                #     t_EEG = eegs[i]
                #     EEG[pos:(pos + len(t_EEG)), :] = t_EEG
                #     pos += len(t_EEG)
                EEG = np.concatenate(eegs)
                print(f"\nTotal number of trials: {np.sum(np.array(event_type) == Event.START)}")
                print(f" - Both feet: {np.sum(np.array(event_type) == Event.BOTH_FEET)}")
                print(f" - Both hands: {np.sum(np.array(event_type) == Event.BOTH_HANDS)}")
                print(f" - Rest: {np.sum(np.array(event_type) == Event.REST)}")
                print(f"{np.sum(np.array(event_type) == Event.HIT)} Hits, {np.sum(np.array(event_type) == Event.MISS)} misses")

                subj = {'eeg': EEG, 
                        'triggers': {'pos': event_pos, 'type': event_type},
                        'session_delims': session_delims}
                savemat(os.path.join(OUT_PATH, f"dataset_{split}.mat"), {'subj': subj})

def check_sanity(eeg,header):
    event_pos = header['EVENT']['POS']
    event_type = header['EVENT']['TYP']

    if sum(event_type == Event.CONT_FEEDBACK) == 0: # no trials found
        return False
    
    if len(event_pos) != len(event_type): # Some information has been lost during saving/loading v
        return False

    if sum(event_type == Event.CONT_FEEDBACK) != (sum(event_type == Event.HIT) +
                                          sum(event_type == Event.MISS) +
                                          sum(event_type == Event.TIMEOUT)): # Trial starts and trial ends are in different numbers
        return False

    start_times = event_pos[event_type == Event.CONT_FEEDBACK]
    end_times = event_pos[(event_type == Event.HIT) |
                         (event_type == Event.MISS) |
                         (event_type == Event.TIMEOUT)]
    diff = end_times - start_times
    if any(diff < 0): # Trial starts and trial ends are misaligned in time
        return False

    return True

def read_mat(spath):
        
    raw = loadmat(spath)
    t_eeg = raw['s']
    t_header = fix_mat(raw['h'])
    return t_eeg,t_header

def read_gdf(spath):
    raw=mne.io.read_raw_gdf(spath,verbose='error')
    events,names = mne.events_from_annotations(raw,verbose='error')
    names = {v:int(k) for k,v in names.items()}
    events_pos = events[:,0]
    events_typ = events[:,2]
    events_typ = [names[e] for  e in events_typ]
    t_eeg = raw.get_data().T

    t_header = {'SampleRate':raw.info['sfreq'],
                'EVENT':{'POS':np.array(events_pos),'TYP':np.array(events_typ)}
                }
    return t_eeg,t_header

if __name__=='__main__':
    main()