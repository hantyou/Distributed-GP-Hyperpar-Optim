function [F_true,range,reso] = standard_field_loading(range,reso,realDataSet,theta)
%STANDARD_FIELD_LOADING Summary of this function goes here
%   Detailed explanation goes here

%% Input interface
disp("%%%%%%%%%%%%%DATA LOAD SETTING%%%%%%%%%%%%%%%")
switch nargin
    case 0
        range=[-1 1;-1 1]; disp('Use default range [-1 1] X [-1 1].');
        reso=[256;256]; disp('Use default resolusion 256 X 256.');
        realDataSet=0; disp('Use default artificial dataset.');
        theta=[5;1;1]; disp('Use default hyperparameter set [5;1;1]');
    case 1
        reso=[256;256]; disp('Use default resolusion 256 X 256.');
        realDataSet=0; disp('Use default artificial dataset.');
        theta=[5;1;1]; disp('Use default hyperparameter set [5;1;1]');
    case 2
        realDataSet=0; disp('Use default artificial dataset.');
        theta=[5;1;1]; disp('Use default hyperparameter set [5;1;1]');
    case 3
        if realDataSet==0
            theta=[5;1;1]; disp('Use default hyperparameter set [5;1;1]');
        else
            disp('Getting real dataset, no hyperparameter set needed.');
        end
end
disp("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

%%
if realDataSet==1
    disp('This exp is down with real dataset GHRSST loaded')
%     loadRealDataset
    [F_true,range,reso] = loadGHRSST(1);

elseif realDataSet==0
    disp('This exp is down with artificial dataset loaded')
    [F_true,reso,range]=loadDataset(1,reso,range,theta);
end
end

