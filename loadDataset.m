function [F_true,reso,range] = loadDataset(select,reso,range,theta)
%LOADDATASET Summary of this function goes here
%   Detailed explanation goes here
if nargin==1
    switch select
        case -1
            disp('Read already generated GP data')
        case -2
            disp('Read field generated according to some function')
        otherwise
            error('Not enough input')
    end
    
elseif nargin==2
    switch select
        case 1
            disp('Generate GP according to given GP_hyperpar')
            error('You have not yet decide the range of field and GP hyperparameters')
        case -1
            disp('Read already generated GP data')
        case 2
            disp('Generate field according to some function')
            error('more info needed for that function')
        case -2
            disp('Read field generated according to some function')
        otherwise
            error('Wrong selection/input')
    end
elseif nargin==3
    switch select
        case 1
            disp('Generate GP according to given GP_hyperpar')
            disp('You have not yet decide the GP hyperparameters')
            disp('sigma_f=5, l=[1,1] will be used')
            theta=[5,1,1]';
        case 2
            disp('Generate field according to some function')
        otherwise
            error('Wrong input/input')
    end
else
    switch select
        case 1
            disp('Generate GP according to given GP_hyperpar')
        case 2
            disp('Not generating GP field, so hyperparameters not used')
        otherwise
            error('Wrong input/input')
    end
end

reso_m=reso(1);
reso_n=reso(2);
range_x1=range(1,:);
range_x2=range(2,:);
if select == 1
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Use below lines to generate a predefined GP
    % Note that l can not be larger than field range
    [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));
    %The parameter of generated GP
    
    l=theta(2:3);
    sigma_f=theta(1);
    tic
    F_true=gen2D_GP(reso_m,reso_n,l,sigma_f,range_x1,range_x2)';
    toc
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Use below lines to save generated GP data set
    save('generatedGPfield.mat','reso_m','reso_n','l','sigma_f','F_true','mesh_x1','mesh_x2');
end


if select == 2
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Use below lines to generate a predefined function
    reso_m=256;
    reso_n=256;
    [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));
    mesh_x=[vec(mesh_x1)'/1.5;vec(mesh_x2)'];
    F_true=20*vecnorm(mesh_x)'.*exp(-vecnorm(mesh_x)'/1);
    F_true=reshape(F_true,reso_m,reso_n);
    % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Use below lines to save predefined function data set
    save('predefinedFunctionField.mat','reso_m','reso_n','F_true','mesh_x1','mesh_x2');
end
% % % % % % % % % % % % % % % % % % % % % % % % % %
% Use below lines to load predefined function data set
if select==-2
    load('predefinedFunctionField.mat');
    
elseif select==-1
    load('generatedGPfield.mat');
end
reso=[reso_m,reso_n];
end

