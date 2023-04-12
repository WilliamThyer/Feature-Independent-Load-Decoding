subs = {'01','02','03','05','07','08','11','12','13','14','15','16','17'};
experiment = 'B01';
numsubs= length(subs);
root = '.';
destination = ['..\analysis\data\',experiment,'\'];
cd(root)
eeglab

for isub = 1:numsubs

    sub_folder = ['..\raw_data\',subs{isub},'\'];
    files = dir(sub_folder);

    unchecked_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_',subs{isub},'_unchecked.set'];

    EEG = pop_loadset(unchecked_file);

    %Titles
    title = EEG.setname;
    xdata_filename = strjoin({destination, title, '_xdata.mat'},'');
    ydata_filename = strjoin({destination, title, '_ydata.mat'},'');
    idx_filename = strjoin({destination, title, '_artifact_idx.mat'},'');
    behavior_filename = strjoin({destination, title, '_behavior.csv'},'');
    info_filename = strjoin({destination, title, '_info.mat'},'');
    % Artifact IDX
    artifact_idx = ~logical(EEG.reject.rejmanual);
    save(idx_filename,'artifact_idx')

    % XData
    EEG = pop_rejepoch(EEG,EEG.reject.rejmanual,0);
    [xdata,chan_idx] = create_xdata(EEG);
    save(xdata_filename, 'xdata');
    
    % YData
    ydata = create_ydata(EEG, xdata);
    save(ydata_filename, 'ydata');
    
    % Info
    unique_ID_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_0',subs{isub},'_info.json'];
    [unique_id,chan_labels,chan_x,chan_y,chan_z,sampling_rate,times] = create_info(EEG, unique_ID_file,chan_idx);
    save(info_filename,'unique_id','chan_labels','chan_x','chan_y','chan_z','sampling_rate','times');
    
    % Save copy of behavior csv
     behavior_file = ['..\raw_data\',experiment,'\',subs{isub},'\',experiment,'_',subs{isub},'.csv'];
     copyfile(behavior_file,behavior_filename);
    
    clear labels num_trials templabel x y checked_trials 
end
disp("DATA EXTRACTION COMPLETE")

function [xdata, chan_idx] = create_xdata(EEG)
    % create xdata for saving to .mat

    num_chans = EEG.nbchan;
    all_chans = strings(num_chans,1);
    for chan = 1:num_chans
        all_chans(chan,:) = EEG.chanlocs(chan).labels;
    end
    chan_idx = ismember(all_chans,{'L-GAZE-X','L-GAZE-Y','R-GAZE-X','R-GAZE-Y','StimTrak','HEOG','VEOG','TP9','GAZE_X','GAZE_Y'});

    xdata = EEG.data(~chan_idx,:,:);
end

function [ydata] = create_ydata(EEG, xdata)
    % create xdata for saving to .mat. This will definitely change based on
    % your portcode structure!

    num_trials = size(xdata,3);
    ydata = zeros(num_trials,1);
    for x=1:num_trials
        sorted_labels = sort(EEG.epoch(x).eventbinlabel);
        char_labels = char(sorted_labels(end));
        ydata(x,:) = str2double(char_labels(5:6));
    end
end

function [unique_id,chan_labels,chan_x,chan_y,chan_z,sampling_rate,times] = create_info(EEG, unique_ID_file, chan_idx)

    % Gather info variables
    chan_labels = {EEG.chanlocs.labels}';
    chan_labels = char(chan_labels(~chan_idx));
    chan_x = [EEG.chanlocs.X];
    chan_y = [EEG.chanlocs.Y];
    chan_z = [EEG.chanlocs.Z];
    chan_x = chan_x(~chan_idx(1:31));
    chan_y = chan_y(~chan_idx(1:31));
    chan_z = chan_z(~chan_idx(1:31));
    sampling_rate = EEG.srate;
    times = EEG.times;
    
    val = jsondecode(fileread(unique_ID_file));
    unique_id = str2double(val.UniqueSubjectIdentifier);
end