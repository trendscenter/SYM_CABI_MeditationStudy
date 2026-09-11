function eeg_preprocess(SUBJECT_IDS)
% add the below 4 lines above this function for debugging
%clear; close all; clc;
%SUBJECT_IDS=[1:8, 10:23];
%SUBJECT_IDS=[1]; %Changed for testing

%eeg_preprocess_1(SUBJECT_IDS)


%To run on server; try
% matlab -batch "eeg_preprocess" OR
% matlab -nodesktop -nosplash -r "run('eeg_preprocess.m'); quit;"

runtype='local' %or could be 'server'
runtype='server' %or could be 'server'

if strcmp(runtype,'local')
    base_dir='/Users/sbasodi1/GSU Dropbox Dropbox/Sunitha Basodi (sbasodi1)/snt_mac/workplace/sym_cabi/SYM_CABI_MeditationStudy/DataPreprocessing/'
    working_dir= [base_dir 'matlab_scripts/']
    % add path of the EEGLAB - add only the roor dir to the addpath, not with subdirectories
    addpath('/Users/sbasodi1/Documents/MATLAB/CustomToolBoxes/eeglab2025.0.0');

    %add gift and SPM12 toolboxes
    % addpath(genpath('/Users/sbasodi1/GSU Dropbox Dropbox/Sunitha Basodi (sbasodi1)/snt_mac/software/MATLAB/Toolboxes/gift-master/GroupICAT/icatb/'));

    % Do not add "trends_tweaked_maca64_spm12" in path as it causes cleanline() to fail : spm12 betapdf error lgamma not found
    % addpath(genpath('/Users/sbasodi1/GSU Dropbox Dropbox/Sunitha Basodi (sbasodi1)/snt_mac/software/MATLAB/Toolboxes/trends_tweaked_maca64_spm12/spm12/'));
    % To install SPM12 on mac with ARM64, download from :  https://trends-public-website-fileshare.s3.amazonaws.com/public_website_files/software/misc/appleSiliconCompatible/spm12.zip
    % Download latest gift code from git repo on master branch (not from
    % releases): https://github.com/trendscenter/gift/tree/master
    % Run: the following command inside icatb gift folder and also from spm12
    % find . -iname "*.mexmaca64" -exec xattr -d com.apple.quarantine {} \;
elseif strcmp(runtype,'server')
    base_dir='/data/users3/sbasodi1/workplace/sy_meditation/eeg_preprocessing/';

    %TODO:
    working_dir=[base_dir 'matlab_scripts/'];

    addpath('/data/users3/sbasodi1/workplace/sy_meditation/eeg_preprocessing/softwares/eeglab2025.0.0');
    %addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/GroupICATv4.0c'));
    %addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/GroupICAT'))
    %addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/spm12'))

    % Install necessary plugins -- sometimes missing on the server
    % Define the EEGLAB plugin directory
    eeglab_path = fileparts(which('eeglab.m'));
    target_dir = fullfile(eeglab_path, 'plugins');

    %List of plugins to install (Name, Version, URL) -- need only once
    plugins = { ...
        'bva-io',   '1.73', 'https://sccn.ucsd.edu/eeglab/plugins/bva-io1.73.zip'; ...
        'cleanline', '2.1', 'https://sccn.ucsd.edu/eeglab/plugins/Cleanline2.1.zip'; ...
        'fmrib',   '2.1',  'https://sccn.ucsd.edu/eeglab/plugins/fMRIb2.1.zip' ...
    };

    % for i = 1:size(plugins, 1)
    %     name = plugins{i, 1};
    %     version = plugins{i, 2};
    %     url = plugins{i, 3};
    %     zip_file = fullfile(target_dir, [name '.zip']);
    %
    %     plugin_check = dir(fullfile(target_dir, [name '*']));
    %     if isempty(plugin_check)
    %         fprintf('Plugin "%s" missing.Downloading %s v%s...\n', name, version);
    %         try
    %             % 1. Download
    %             websave(zip_file, url);
    %
    %             % 2. Unzip
    %             fprintf('Installing %s...\n', name);
    %             unzip(zip_file, target_dir);
    %
    %             % 3. Cleanup
    %             % delete(zip_file);
    %             fprintf('%s installed successfully.\n\n', name);
    %         catch ME
    %             fprintf('Failed to install %s: %s\n', name, ME.message);
    %             %rethrow ME
    %         end
    %     else
    %         fprintf('Plugin "%s" already exists in %s. Skipping.\n', name, plugin_check(1).name);
    %     end
    % end

    % Update MATLAB path and EEGLAB internal list
    [ALLEEG, EEG, CURRENTSET] = eeglab('nogui'); %This seems to work better than the below three commands
    cleanline_path = fullfile(eeglab_path, 'plugins', 'cleanline'); % Or your specific folder name
    % 2. Add the plugin and ALL its subfolders (including 'external' where hlp functions live)
    if exist(cleanline_path, 'dir')
        addpath(genpath(cleanline_path));
        rehash;
        fprintf('CleanLine paths refreshed.\n');
    else
        error('CleanLine folder not found at %s', cleanline_path);
    end

    %addpath(genpath(target_dir));
    %rehash;
    %eeg_checkextensions;


else
   error('Error. \n runtype must be ''local'' or ''server''.')
end

base_data_dir = fullfile(fileparts(myPath), 'data');
raw_data_dir = fullfile(base_data_dir, 'raw_all');
output_preprocessed_dir = fullfile(base_data_dir, 'preprocessed_eeg');
%create output dir
mkdir(output_preprocessed_dir);
id_tag = strjoin(string(SUBJECT_IDS), '-');
log_file = fullfile(output_preprocessed_dir, sprintf('proprocessing_log_SUBID_%s_date_%s.txt',id_tag, string(datetime('today'))));
% Open the file for appending ('a')
log_file_id = fopen(log_file, 'a');

fprintf(log_file_id, '\n\n#######################\n\n');
fprintf(log_file_id, '%s: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'),'Started EEG Preprocessing.');

%EEG file types
EEG_FILENAME_MAPPING = containers.Map({'6_SlowBreathingSYMTask', 'CABI_SYM_AversiveVisualStimuliTask', 'SYM_CABI_MeditativeState', 'SYM_CABI_RSfMRI_1', 'SYM_CABI_RSfMRI_2'}, ...
    {'4-SlowBreathingSYMTask', '2-A versiveVisualStimuliTask', '5-MeditativeState','1-RSfMRI_1', '3-RSfMRI_1'});


num_channels_in_cap=64;
eeg_format_file = 'vhdr'; % Select one of {'vhdr', 'set'}
NON_EEG_CHANNEL_NUMBERS = [32]; %Channel names like ECG, EOG*
ECG_CHANNEL_NUMBER = 32;
SCANNER_TRIGGER_ID='X  1';

num_valid_subjects=length(SUBJECT_IDS);
subject_dirnames=strings(1, num_valid_subjects);
for indx = 1:num_valid_subjects
    subject_dirnames(indx) = sprintf('SUB%02d', SUBJECT_IDS(indx));
end

if strcmp(eeg_format_file, 'vhdr')
  for indx  = 1:num_valid_subjects
    subject_dirname = subject_dirnames(indx);
    subject_data_dir=fullfile(raw_data_dir, subject_dirname);
    %if we dont cast them to char() pop_saveset() internally converting some of them to string arrays and gives error.

    subject_output_dir = char(fullfile(output_preprocessed_dir, subject_dirname));

    %create output directory for subject
    mkdir(subject_output_dir);

    %find existing eeg files for the subject
    files_EEG = dir(fullfile(raw_data_dir, subject_dirname, '**', '*vhdr'));   % for loading .vhdr files
    fprintf(log_file_id, '\n\nStarting to preprocess EEG data of %s \n', subject_dirname)

    for f = 1:length(files_EEG)
        try
            if strcmp(runtype,'local')
                [ALLEEG, EEG, CURRENTSET] = eeglab;
            elseif strcmp(runtype,'server')
                [ALLEEG, EEG, CURRENTSET] = eeglab('nogui');
            end
            fprintf(log_file_id, '\n\nCurrent file: %s \n', files_EEG(f).name)

            fname  = extractBefore(files_EEG(f).name, '.vhdr');   % for .vhdr files
            %find output filename
            splitStrings = split(fname, "_SUB");
            out_filename = char(EEG_FILENAME_MAPPING(string(splitStrings(1))))


            EEG = pop_loadbv(files_EEG(f).folder, files_EEG(f).name);    % load the .vhdr file
            %EEG = pop_loadbv('/Users/sbasodi1/GSU Dropbox Dropbox/Sunitha Basodi (sbasodi1)/snt_mac/workplace/sym_cabi/SYM_CABI_MeditationStudy/DataPreprocessing/data/raw_all/SUB01/SUB_001_12182024_EEG/', '6_SlowBreathingSYMTask_SUB01.vhdr', [1 2251600], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]);

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

            % 1. REMOVE scanner artifact
            disp('****EPI Artifact Removal')

            % Using FMRIB FASTR toolbox
            cleaned_filename=[fname '_cleaned_pop_fmrib_fastr.set']; %set file
            % Check if the file exists
            if isfile(fullfile(subject_output_dir, cleaned_filename))
                EEG = pop_loadset('filename', cleaned_filename, 'filepath', subject_output_dir);
            else
                % Handle the absence of the file (e.g., create it, show error)
                %EEG = pop_fmrib_fastr(EEG, 0, 1, 10, SCANNER_TRIGGER_ID, 1, 0, 0, 0, 0, 0.03, NON_EEG_CHANNEL_NUMBERS, 'auto');
                %EEG = pop_fmrib_fastr(EEG, 0, 10, 30, SCANNER_TRIGGER_ID, 1,0,0,0,0,0.03,NON_EEG_CHANNEL_NUMBERS,'auto');
                EEG = pop_fmrib_fastr(EEG, 0, 4, 10, SCANNER_TRIGGER_ID, 1,0,0,0,0,0.03,NON_EEG_CHANNEL_NUMBERS,'auto');
                % TODO Uncomment after testing above line
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
                EEG = pop_saveset(EEG, 'filename', cleaned_filename, 'filepath', subject_output_dir);
                % nosure if works[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'savenew', fullfile(subject_output_dir, cleaned_filename), 'gui', 'off');
                % %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
            end


            % resample to 1kHz
            disp('***Resampling')
            EEG = pop_resample( EEG, 1000);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');


            % 2. cardiac artifacts
            disp('***Identifying and Removing Cardiac Artifacts')

            cleaned_filename=[fname '_cleaned_and_qrs.set'];
            if isfile(fullfile(subject_output_dir, cleaned_filename))
                EEG = pop_loadset('filename', cleaned_filename, 'filepath', subject_output_dir);
            else
                try
                     EEG = pop_fmrib_qrsdetect(EEG, ECG_CHANNEL_NUMBER, 'qrs', 'no');    % replace your ECG channel number
                     [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
                catch ME
                    fprintf(log_file_id, 'QRS Detection failed for this file. Check ECG signal quality.\n');
                    fprintf(log_file_id, 'Error message: %s\n', ME.message);
                    % You can choose to skip this file or save it as 'failed_qrs.set'
                    rethrow(ME);
                end

                EEG = pop_fmrib_pas(EEG,'qrs', 'obs',3);  % this stage takes a while. try 'obs' too!
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');
                EEG = pop_saveset(EEG, 'filename', cleaned_filename, 'filepath', subject_output_dir);

            end

            %%% qrs doesn't work in some cases that includes all the epoch, warning
            % see 'eeg_preprocess_tom_eyesopen'
            % cut off the first N epoch
            % disp('epoch removing')
            % EEG = pop_select( EEG, 'trial', [2:256] );% [...] contains the channel range, i.e. 31 is EOG, 32 is ECG
            % [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');  % overwrite in memory


            % Removing line noise
            disp('***Removing line noise')
            EEG = pop_cleanline(EEG, 'bandwidth',2, 'chanlist',[1:num_channels_in_cap] , 'computepower',1, 'linefreqs',60, 'newversion',0, 'normSpectrum',0, 'p',0.01, 'pad',2, 'plotfigures',0, 'scanforlines',0, 'sigtype','Channels', 'taperbandwidth',2, 'tau',100, 'verb',1, 'winsize',4, 'winstep',1);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');

            % TODO- CHECK WITH SOUVIK IF THIS STEP SHOULD BE HERE
            % remove channels (EOG, ECG) - might be different in later subjects
            disp('***Removing Channels')
            EEG = pop_select( EEG, 'rmchannel',{'ECG','EOGL','EOGU'});
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');  % overwrite in memory

            % Filtering the EEG
            disp('***Filtering the EEG')
            EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 30);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'overwrite','on','gui','off');

              % common average reference
            disp('***common average reference')
            EEG = pop_reref( EEG, [], 'refstate', 0);
            [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');

            EEG = eeg_checkset( EEG );
            EEG = pop_rmbase( EEG, [ ]); % new baseline
            [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
            %pop_saveset(EEG,'6');


            % run ICA
            disp('***Running ICA')
            EEG = pop_runica(EEG,  'icatype', 'runica', 'dataset', 1, 'options', { 'extended', 1});% standard ICA = 0, extended = 1; take a while!
            [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
            save(fullfile(subject_output_dir, [out_filename,'_ICA_processed_without_filtering.mat']), 'EEG');
            pop_saveset(EEG, [out_filename,'_ICA_processed_without_filtering.mat'], subject_output_dir);
            % eeglab redraw;
            fprintf(log_file_id, 'Finished generating ICA .mat for %s : %s\n\n', subject_dirname, out_filename)
            disp('***Finished generating ICA')

        catch ME
            fprintf(log_file_id, 'Error in preprocessing file %s : %s\n', subject_dirname, files_EEG(f).name)
            fprintf(log_file_id, 'Error message: %s\n', ME.message);

        end
    end
  end
elseif strcmp(eeg_format_file, 'set')
    files_EEG = files_EEG(~startsWith({files_EEG.name}, '.set'));  % Remove hidden files
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


% Close the file
fclose(log_file_id);
exit;
end