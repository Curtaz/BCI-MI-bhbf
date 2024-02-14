clearvars; clc;
%%

% DATA_PATH = "C:\Users\tomma\Documents\Uni\PhD\data\Neurorobotics-data\2023trainval\";
DATA_PATH = "C:\Users\tomma\Documents\Uni\PhD\data\cBCI\all_subjects\";
DATA_PATH = "C:\Users\tomma\Documents\Uni\PhD\data\cBCI\sbj4_tr_rigorous\";
OUT_PATH = strcat(DATA_PATH,"preprocessed/");
FILT_ORDER = 2;
FC_l = 40;
FC_H = 2;
NUM_CHANNELS = 32;
LAP_PATH = "C:\Users\tomma\Documents\Uni\PhD\code\cBCI-MI-bhbf\config\lapmask_antneuro_32.mat";


WINSHIFT = 0.0625; %[s]
WINSIZE = 1; %[s]
Filenames = containers.Map (["train","val","test"],{ dir(strcat(DATA_PATH,"train/*.mat"));dir(strcat(DATA_PATH,"val/*.mat"));dir(strcat(DATA_PATH,"test/*.mat"));});
global EVENT
EVENT.START = 1;
EVENT.RIGHT_HAND = 769;
EVENT.LEFT_HAND = 770;
EVENT.BOTH_HANDS = 773;
EVENT.BOTH_FEET = 771;
EVENT.CONT_FEEDBACK = 781;
EVENT.REST = 783;
EVENT.FIXATION = 786;
EVENT.HIT = 897;
EVENT.MISS = 898;
EVENT.TIMEOUT = 899;

%%
if ~exist(OUT_PATH, 'dir')
   mkdir(OUT_PATH)
end

load(LAP_PATH,'lapmask');
for split = ["train" "val" "test"]
    printf("%s: reading from %s\n", split, strcat(DATA_PATH,split))
    eegs = cell(length(Filenames(split)),1);
    session_delims = [];
    event_pos = [];
    event_type = [];
    settings = eegc3_smr_newsettings();

    last_eeg_len = 0;
    dim_tot = 0;
    i=1;
    for file=Filenames(split)'
        disp(file.name)

        load(char(strcat(DATA_PATH,split,'/',file.name)),'s','h');
        t_eeg = s;
        t_header = h;
        if check_sanity(t_eeg,t_header)
            %% Read data, setup configuration
            events = t_header.EVENT;
    
            fs = t_header.SampleRate;
            winShift = fs*WINSHIFT;
            winSize = fs * WINSIZE;
            %% Preprocessing steps
            
            % Apply Laplacian spatial filter
            t_eeg = t_eeg(:,1:NUM_CHANNELS) * lapmask;
            % Apply bandpass filter
            for ch = 1:NUM_CHANNELS
                [b,a] = butter(FILT_ORDER,2*FC_l/fs,"low");
                t_eeg(:,ch) = filter(b,a,t_eeg(:,ch)); 
            
                [d,c] = butter(FILT_ORDER,2*FC_H/fs,"high");
                t_eeg(:,ch) = filter(d,c,t_eeg(:,ch)); 
            end

            eegs{i} = t_eeg;
            event_pos = [ event_pos; (t_header.EVENT.POS(:) + dim_tot)];
            event_type = [ event_type; (t_header.EVENT.TYP(:))];
            session_delims = [ session_delims; (last_eeg_len)+length(t_eeg)];
            dim_tot = dim_tot + length(t_eeg);
            i = i+1;
        else
            printf("Bad format, corrupt file or whatever, skipping..\n")
        end

    end
    %%
    EEG = nan(dim_tot, NUM_CHANNELS);
    pos = 0;
    for i=1:length(eegs)
        t_EEG = eegs{i};
        EEG(pos+(1:length(t_EEG)),:) = t_EEG;
        pos = pos+length(t_EEG);
    end

    printf("Total number of trials: %d\n", sum(event_type==EVENT.START));
    printf(" - Both feet: %d\n",sum(event_type==EVENT.BOTH_FEET));
    printf(" - Both hands: %d\n",sum(event_type==EVENT.BOTH_HANDS));
    printf(" - Rest: %d\n",sum(event_type==EVENT.REST));
    printf("%d Hits, %d misses\n",sum(event_type==EVENT.HIT),sum(event_type==EVENT.MISS));

    subj.settings = settings;
    subj.eeg = EEG;
    subj.triggers.pos = event_pos;
    subj.triggers.type = event_type;
    subj.session_delims = session_delims;
    save(strcat(OUT_PATH,"dataset_",split,".mat"),'subj','-v7.3');
end

%%

mask = event_type == EVENT.START | ...
       event_type == EVENT.BOTH_FEET |...
       event_type == EVENT.BOTH_HANDS |...
       event_type == EVENT.HIT |...
       event_type == EVENT.MISS;



function [sanity] = check_sanity(eeg,header)
    event_pos = header.EVENT.POS;
    event_type = header.EVENT.TYP;
    global EVENT
    if length(event_pos) ~= length(event_type)
        sanity = false;
        return
    end
    if sum(event_type==EVENT.START) ~= (sum(event_type==EVENT.HIT)+ ... 
                                        sum(event_type==EVENT.MISS)+ ...
                                        sum(event_type==EVENT.TIMEOUT))
        sanity = false;
        return
    end
    start_times = event_pos(event_type==EVENT.START);
    end_times = event_pos(event_type==EVENT.HIT | ...
                          event_type==EVENT.MISS |...
                          event_type==EVENT.TIMEOUT);
    diff = end_times -start_times;
    if any(diff<0)
        sanity=false;
        return
    end
    sanity=true;
end
%% Other
% Show EEG
% for i = 1:16
%     subplot(16,1,i)
%     plot(EEG(5000:length(EEG),i))
% end
% shg

%%Filtro 2-40hz
%%windows size 128
%%Sample size 32
%%overlap 0.5 s

