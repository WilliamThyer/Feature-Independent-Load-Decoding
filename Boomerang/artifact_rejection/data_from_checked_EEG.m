subs = {'12','13','14','15','16','17','18','19'};

experiment = 'B01';
numsubs= length(subs);
root = '.';
destination = ['..\analysis\data\',experiment,'\'];
cd(root)
% eeglab

for isub = 1:numsubs
        checked_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_',subs{isub},'_checked.set'];
        EEG = pop_loadset(checked_file);
        
        %Titles
        title = [experiment, '_', EEG.setname(end-1:end)];
        xdata_filename = [destination, title, '_xdata.mat'];
        ydata_filename = [destination, title, '_ydata.mat'];
        idx_filename = [destination, title, '_artifact_idx.mat'];
        behavior_filename = [destination, title, '_behavior.csv'];
        info_filename = [destination, title, '_info.mat'];
        
        % Remove unwanted channels and save xdata
        num_chans = EEG.nbchan;
        all_chans = strings(num_chans,1);
        for chan = 1:num_chans
            all_chans(chan,:) = EEG.chanlocs(chan).labels;
        end
        chan_idx = ismember(all_chans,{'L_GAZE_X','L_GAZE_Y','R_GAZE_X','R_GAZE_Y','StimTrak','HEOG','VEOG','TP9','GAZE_X','GAZE_Y'});

        xdata = EEG.data(~chan_idx,:,:);
        save(xdata_filename, 'xdata');
        
        % Extract and save
        num_trials = size(xdata,3);
        ydata = zeros(num_trials,1);
        for x=1:num_trials
            sorted_labels = sort(EEG.epoch(x).eventbinlabel);
            char_labels = char(sorted_labels(end));
            ydata(x,:) = str2double(char_labels(5:6));
        end
        
        save(ydata_filename, 'ydata');
        
        % Gather info variables
        chan_labels = {EEG.chanlocs.labels}';
        chan_labels = char(chan_labels(~chan_idx));
        chan_x = [EEG.chanlocs.X];
        chan_y = [EEG.chanlocs.Y];
        chan_z = [EEG.chanlocs.Z];
%         chan_x = chan_x(~chan_idx);
%         chan_y = chan_y(~chan_idx);
%         chan_z = chan_z(~chan_idx);
        chan_x = chan_x(~chan_idx(1:31));
        chan_y = chan_y(~chan_idx(1:31));
        chan_z = chan_z(~chan_idx(1:31));
        sampling_rate = EEG.srate;
        times = EEG.times;
        
        unique_ID_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_0',subs{isub},'_info.json'];
        val = jsondecode(fileread(unique_ID_file));
        unique_id = str2double(val.UniqueSubjectIdentifier);

        save(info_filename,'unique_id','chan_labels','chan_x','chan_y','chan_z','sampling_rate','times');
        
        %Saving artifact index for indexing behavior file
        num_rows = size(EEG.event,2);
        all_trials = zeros(num_rows,1);
        for x = 1:num_rows
            all_trials(:,x) = EEG.event(x).bepoch;
        end
        checked_trials = unique(all_trials);
        
        unchecked_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_',subs{isub},'_unchecked.set'];
        EEG = pop_loadset(unchecked_file);
        unchecked_trials = [1:EEG.trials]';
        artifact_idx = ismember(unchecked_trials,checked_trials);
        
        save(idx_filename,'artifact_idx')
        
        % Save copy of behavior csv
         behavior_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_',subs{isub},'.csv'];
         copyfile(behavior_file,behavior_filename);
        
        clear labels num_trials templabel x y checked_trials 
end
"DATA EXTRACTION COMPLETE"