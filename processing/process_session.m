clearvars; clc;
%%

% DATA_PATH = "C:\Users\tomma\Documents\Uni\PhD\data\Neurorobotics-data\2023trainval\";
DATA_PATH = 'C:\Users\tomma\Downloads\test\g1.20240115.153338.gdf';
OUT_PATH = "C:\Users\tomma\Downloads\test\g1.20240115.153338_filtered.mat";
FILT_ORDER = 2;
FC_l = 40;
FC_H = 2;
NUM_CHANNELS = 32;

settings = eegc3_smr_newsettings();

[eeg,h] = sload(DATA_PATH);
eeg = eeg(:,1:NUM_CHANNELS);
eeg_filtered = copyaxis(eeg);

eeg_filtered = eeg_filtered(:,1:NUM_CHANNELS) * settings.modules.smr.laplacian;
fs = h.SampleRate;
% Apply bandpass filter
for ch = 1:NUM_CHANNELS
    [b,a] = butter(FILT_ORDER,2*FC_l/fs,"low");
    eeg_filtered(:,ch) = filter(b,a,eeg_filtered(:,ch)); 

    [d,c] = butter(FILT_ORDER,2*FC_H/fs,"high");
    eeg_filtered(:,ch) = filter(d,c,eeg_filtered(:,ch)); 
end

save(OUT_PATH,'eeg')
