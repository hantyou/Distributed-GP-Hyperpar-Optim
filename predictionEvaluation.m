clc,clear;
%% Data loading
dataSourceOption=1; 
%1: Inherit from HPerpar Optimz part with files name as "workspaceForDebug.mat"
%2: Freshly generated data

if dataSourceOption==1
    try
        load('workspaceForDebug.mat')
    catch
        disp("data is not ready")
        return;
    end
    
    samplingMethod; 
    % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
    if samplingMethod==1
        disp("Sampling method: uniformly distirbuted accross region;")
    elseif samplingMethod==2
        disp("Sampling method: unear agents position, could lose some points if out of range;")
    end
    
    % The sampling positions are stored in X,
    % corresponding clean values in Y,
    % noisy values in Z.
    
elseif dataSourceOption==2
    % If generate new data, indicate some options
    samplingMethod=2;
    
    range_x1=[-5,5];
    range_x2=[-5,5];
    range=[range_x1;range_x2];
    
end
