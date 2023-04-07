function [EEG] = align_channels(EEG)
    %%%
    % Realign channels for inspection. 
    %%%
    
    eog_idx = ismember({EEG.chanlocs.labels},{'EOG','VEOG','HEOG'});
    EEG.data(eog_idx,:,:) = EEG.data(eog_idx,:,:) - 400;

    eye_idx = ismember({EEG.chanlocs.labels},{'L-GAZE-Y','R-GAZE-Y','L-GAZE-X','R-GAZE-X'});
    EEG.data(eye_idx,:,:) = EEG.data(eye_idx,:,:) + 200;
    
    stim_idx = ismember({EEG.chanlocs.labels},{'StimTrak'});
     EEG.data(stim_idx,:,:) = EEG.data(stim_idx,:,:) + 300;
    
end

