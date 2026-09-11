clear; close all; clc;
% add path of the EEGLAB
addpath ('/Users/admin-sphadikar/Dropbox (GSU Dropbox)/Postdoc-GSU/Toolboxes/eeglab2024.1');   % check your EEG virsion and correct accordingly
% eeglab;

ddir = '/Volumes/Souvik/PostDoc@GSU/64channel_eeg_fmri/EEG_raw/';    % All data directory
pdir = '/Volumes/Souvik/PostDoc@GSU/64channel_eeg_fmri/EEG_preprocessed';    % Output directory

% files_EEG = dir(fullfile(ddir, '*vhdr'));   % for loading .vhdr files
files_EEG = dir(fullfile(ddir, '*set'));   % for loading .set files
files_EEG = files_EEG(~startsWith({files_EEG.name}, '.'));  % Remove hidden files

for f = 1:length(files_EEG)
    eeglab;
    fprintf('sub %d start \n', f)
    disp('Importing')
    % fname  = extractBefore(files_EEG(f).name, '.vhdr');   % for .vhdr files
    fname  = extractBefore(files_EEG(f).name, '.set');   % for .set files

    % EEG = pop_loadbv(ddir, [files_EEG(f).name]);    % load the .vhdr file
    EEG = pop_loadset(fullfile(ddir, files_EEG(f).name));  % Load the .set file
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', fname, 'gui', 'off');

    % if already converted
    %fname = 'rest0003.set';
    %EEG = pop_loadset(f(dcnt).name);
    %[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'setname', fname, 'gui', 'off');

    % enter channels
    % Select the default channel locations only if the EEG doesn't have
    % them.
    % disp('***Getting channel labels & locations')
    % EEG = pop_chanedit(EEG,  'lookup', '/Users/admin-sphadikar/Dropbox (GSU Dropbox)/Postdoc-GSU/Toolboxes/eeglab2024.1/plugins/dipfit/standard_BESA/standard-10-5-cap385.elp');
    % [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    % %pop_saveset(EEG,'16_ori');

    %%% check epochs (manual)
    %EEG = pop_editeventvals(EEG, 'changefield',{248, 'type', 'R128'});
    %[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

    % epoch the data
    % disp('***Epoching')
    % EEG = pop_epoch( EEG, {'R128'}, [0 2], 'newname', [fname ' epochs'], 'epochinfo', 'yes');% R128 is the usual scanner trigger, should work for most. NB: not all participants have all triggers, so one option is to find the scanner onset and then epoch the TR's by samples
    % EEG = pop_epoch(EEG);

    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
    EEG = eeg_checkset( EEG );
    EEG = pop_rmbase( EEG, [ ]); % new baseline
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    %pop_saveset(EEG,'6');

    % remove scanner artefact
    disp('****EPI Artifact Removal')

    % Using FMRIB FASTR toolbox
    EEG = pop_fmrib_fastr(EEG, 0, 1, 30, 'R128', 1, 0, 0, 0, 0, 0.03, [32, 63, 64], 'auto');
    % change here: find your non-eeg channels and replace the channel numbers ^^^^^^^^
   
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
    % [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

    % resample to 1kHz
    disp('***Resampling')
    EEG = pop_resample( EEG, 1000);

    %%% qrs doesn't work in some cases that includes all the epoch, warning
    % see 'eeg_preprocess_tom_eyesopen'
    % cut off the first N epoch
    % disp('epoch removing')
    % EEG = pop_select( EEG, 'trial', [2:256] );% [...] contains the channel range, i.e. 31 is EOG, 32 is ECG
    % [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');  % overwrite in memory
    
    % cardiac artifacts
    disp('***Identifying and Removing Cardiac Artifacts')
    EEG = pop_fmrib_qrsdetect(EEG, 32, 'qrs', 'no');    % replace your ECG channel number
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
    EEG = pop_fmrib_pas(EEG,'qrs', 'gmean');  % this stage takes a while. try 'obs' too!
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');

    % Removing line noise
    disp('***Removing line noise')
    EEG = pop_cleanline(EEG, 'bandwidth',2, 'chanlist',[1:64] , 'computepower',1, 'linefreqs',60, 'newversion',0, 'normSpectrum',0, 'p',0.01, 'pad',2, 'plotfigures',0, 'scanforlines',0, 'sigtype','Channels', 'taperbandwidth',2, 'tau',100, 'verb',1, 'winsize',4, 'winstep',1);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');

    % remove channels (EOG, ECG) - might be different in later subjects
    disp('***Removing Channels')
    EEG = pop_select( EEG, 'rmchannel',{'ECG','EOGL','EOGU'});
    [ALLEEEEG.nbchanG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');  % overwrite in memory

    % common average reference
    disp('***common average reference')
    EEG = pop_reref( EEG, [], 'refstate', 0);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
    
    % Filtering the EEG
    disp('***Filtering the EEG')
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 30);
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');

    % run ICA
    disp('***Running ICA')
    EEG = pop_runica(EEG,  'icatype', 'runica', 'dataset', 1, 'options', { 'extended', 1});% standard ICA = 0, extended = 1; take a while!
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    pop_saveset(EEG, [fname,'_processed.mat'], pdir);
    % eeglab redraw;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% separate stage from batch running!!%%%%
% % need to MANUALLY select 'relevant/real' ICs and removal
% % or use ICLabel plugin
% pop_selectcomps(EEG, [1:30] );
% [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
% EEG = pop_subcomp(EEG,[1 2 3 5],0); % these are the ones that were removed
% [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
%  pop_saveset(EEG,[fname,'_icaprocessed.set'],pdir);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %clear study
% %ALLCOM = {'[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;'}; LASTCOM = '';STUDY = []; CURRENTSTUDY = 0; ALLEEG = []; EEG=[]; CURRENTSET=[]; eval(LASTCOM); eegh( LASTCOM ); eeglab redraw;


