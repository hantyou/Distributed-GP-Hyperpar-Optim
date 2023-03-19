%% Setting not related to the algorithm itself
clc, clear;
close all;
start_time=datetime('now');
disp(strcat("The code begins at ",datestr(start_time)))
F = findall(0,'type','figure','tag','TMWWaitbar');
delete(F);
try
    opengl software
catch
end
% Turn off figure display if in commandline matlab
if usejava('desktop')
    figureVisibility='on';
else
    figureVisibility='off';
end
set(0,'DefaultFigureVisible',figureVisibility)

clear figureVisibility F
%% Define field size for monitoring
%%%% Many parameters for the simulation are defined here %%%%
rng(100,'twister') % make sure the random sequence is stable
rand(17+16,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Field Properties %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
realDataSet=0;
% 0: use 2D-GP artificial dataset; 1: Use real dataset (Which is loaded by loadRealDataset)
samplingMethod=1;
% 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
inputDim=2;
range_x1=[-5,5];
range_x2=[-5,5];
range=[range_x1;range_x2];
sigma_n=sqrt(0.1); % sampling noise level
realz=[5,1,1,0];
realz(end)=sigma_n;
% defined real hyperparameter if the dataset is 2D-GP. If not, this value
% will be set to zero later.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Agent Properties %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=16; % number of agents
TotalNumLevel=M*50; % Total number of measurement points
everyAgentsSampleNum=floor(TotalNumLevel/M); % Equally distribute measurements to the agents
Agents_measure_range=3; % The range of how far the agent can take measurement

Agents_commuRange=1.2;
Agents_commuRange=3; % The range of how far agents can communicate.
% To generate the same results as in the paper:
% Set to 3 when using 2D-GP dataset;
% Set to 1.2 when using SST dataset;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Experiment Setup %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pass 1 to the method that should be simulated,
% Pass 0 to those who should not.
run_GD=0;
run_ADMM=0;
run_pxADMM=1;
run_ADMM_fd=0;
run_pxADMM_fd_fast_sync=1;
run_pxADMM_fd_fast_async=1;
run_pxADMM_fd_sync=1;
run_pxADMM_fd_async=1;

if realDataSet==0 %initial hyperpar for 2D-GP
    initial_sigma_f=3;
    initial_sigma_n=1;
    initial_l=2*ones(1,inputDim);
else % initial hyperpar for SST
    initial_sigma_f=6;
    initial_sigma_n=0.5;
    initial_l=2*ones(1,inputDim);
end

epsilon = 1e-3; % used for stop criteria
rho_glb=400;
L_glb=4000;
GD_step_size=0.00005; % If gradient descent is simulated, the step size is defined here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dispTxt=fprintf("There will be %d agents in the experiment, a total number of %d measurements will be distributed to them.", M, TotalNumLevel);
disp(dispTxt);
%% parpool setup
% delete(gcp('nocreate'))
if isempty(gcp('nocreate'))
    try
        parpool(M);
    catch
        parpool(8)
    end
end

%% read/generate underlying field, and take samples in the field
reso_m=256; % resolution of field in x direction
reso_n=256; % resolution of field in y direction
reso=[reso_m,reso_n];

if realDataSet==1
    disp('This exp is down with SST dataset loaded')
    packedFieldInfo={M,TotalNumLevel,everyAgentsSampleNum,Agents_measure_range,samplingMethod};
    ncFname="./SSTdata/20220402090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc";
    [pO,newRange]=loadRealDataset(packedFieldInfo,ncFname,1);
    % packedOutput={X,Y,Z,subSize,sampleIdx,sigma_n,agentsPosiY};
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    X=pO{1};Y=pO{2};Z=pO{3};subSize=pO{4};sampleIdx=pO{5};
    sigma_n=pO{6};agentsPosiY=pO{7}; F_true=pO{8}; 
    Agents_Posi=pO{9};clear pO;
    sampleSize=sum(subSize);
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    range_x1=newRange{1}; range_x2=newRange{2};
    reso_m=newRange{3};reso_n=newRange{4}; clear newRange;

    realz=[0,0,0,0]; % real hyperparameters unknown
else
    disp('This exp is down with artificial dataset loaded')
    [F_true,reso]=loadDataset(1,reso,range,realz(1:(end-1)));
    [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Decide sample pointssubSize=ones(M,1)*everyAgentsSampleNum;
    subSize=ones(M,1)*everyAgentsSampleNum;
    Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,M)*0.9;
        unifrnd(range_x2(1),range_x2(2),1,M)*0.9];
    [X,subSize,sampleIdx] = decideSamplePoints(samplingMethod,subSize,range,Agents_Posi,Agents_measure_range);
    sampleSize=sum(subSize);
    X1=X(1,:);
    X2=X(2,:);
    sampleError=randn(1,sampleSize)*sigma_n;
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Take sample
    Y=interp2(mesh_x1,mesh_x2,F_true,X1,X2);
    agentsPosiY=interp2(mesh_x1,mesh_x2,F_true,Agents_Posi(1,:),Agents_Posi(2,:));
    Z=Y+sampleError; % The observation model (measurement model)
end
[mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));

%% Dataset division and agents initialization
% Now the sample points has been taken at given positions in given pattern
% Distribute according to range of detection
localDataSetsSize=subSize;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Create ideces to indicate the samples stored on each agent
idx1=1:sampleSize;
idx=ones(M,max(subSize));
for m=1:M
    idx(m,1:subSize(m))=idx1(sampleIdx(m)+1:sampleIdx(m+1));
end
idxedZ=Z(idx);
subDataSetsZ=idxedZ;
inputDim=size(X,1);
idxedX=zeros(inputDim,max(subSize),M);
for m=1:M
    idxedX(:,:,m)=X(:,idx(m,:));
end
subDataSetsX=idxedX;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Generate agents entities
% The agents properties can be found in file ./agent.m
disp("The sample numbers on the agents are ")
dispTxt="[";
for m=1:M-1
    dispTxt=strcat(dispTxt,sprintf("%d, ",localDataSetsSize(m)));
end
dispTxt=strcat(dispTxt,sprintf("%d]",localDataSetsSize(M)));
disp(dispTxt);

Agents=agent.empty(M,0); % first create empty list of agents
for m=1:M
    Agents(m).Code=m; % Give a sequential code to each agent
    Agents(m).TotalNumLevel=TotalNumLevel; % Let agent know how many samples in total (not used in algorithm)
    Agents(m).Z=subDataSetsZ(m,1:subSize(m))'; % Pass noisy data
    Agents(m).X=subDataSetsX(:,1:subSize(m),m); % Pass sampling positions
    Agents(m).idx=idx(m,1:subSize(m)); % Pass idx of data at original global dataset
    Agents(m).N_m=localDataSetsSize(m); % The size of local dataset
    Agents(m).M=M; % Let the agent know how many agents are working (not used in algorithm)
    Agents(m).NLL=0; % NLL value initialization
    Agents(m).action_status=1; % A value used in true parallel simulation (not used in algorithm)
    Agents(m).realdataset=realDataSet; % Let agents know if they are working on real datasets (not used in algorithm)
    Agents(m).commuRange=Agents_commuRange; % Define the communication ability for agents
    Agents(m).realz=realz; % Pass real hyperparameters as reference for plot
    Agents(m).distX1=dist(Agents(m).X(1,:)).^2; % calculate distance matrix in advance to save computation power
    Agents(m).distX2=dist(Agents(m).X(2,:)).^2; % calculate distance matrix in advance to save computation power
    Agents(m).distXd=zeros(subSize(m),subSize(m),inputDim);
    Agents(m).Position=Agents_Posi(:,m);
    for d=1:inputDim
        Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2; % calculate distance matrix in advance to save computation power
    end
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% decide there to store the simulation results
top_dir_path='results/HO/';
folder_name=strcat(num2str(M),'_a_',num2str(TotalNumLevel),'_pl');
mkdir(top_dir_path,folder_name);
results_dir=strcat(top_dir_path,'/',folder_name);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% delete useless variables
clear idx1 idx idexedZ idexedX subDataSetsX

%% uncomment to generate real underlying NLL map
% theta_range=[[log(3)/log(10),log(7)/log(10)];[log(0.5)/log(10),log(2)/log(10)]];
% LL=generateLikelihoodMap(X,Z,theta_range,sigma_n);
% Xs=cell(M,1);
% Zs=cell(M,1);
% for m=1:M
%     Xs{m}=Agents(m).X;
%     Zs{m}=Agents(m).Z;
% end
% LL2 = generateLikelihoodMap2(Xs,Zs,theta_range,sigma_n);
%% Set topology 
% This section generate topology for the MAS 
Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
A_full=generateTopology(Agents,Topology_method);
for m=1:M
    Agents(m).A=A_full(1:M,1:M);
    Agents(m).Neighbors=find(Agents(m).A(Agents(m).Code,:)~=0);
    Agents(m).N_size=length(Agents(m).Neighbors);
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Check connectivity
G=graph(A_full(1:M,1:M));
L = laplacian(G);
[~,v]=svd(full(L));
v=diag(v);
if v(end-1)==0
    disp("Error: graph not connected")
end
if 1
    gcf=figure('visible','off');
    hold on;
    imagesc(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n),F_true);


    scatter(Agents_Posi(1,:),Agents_Posi(2,:))
    for m=1:M
        for n=m:M
            if A_full(m,n)>0
                posi=[Agents(m).Position,Agents(n).Position];
                Posi_dim=length(Agents(m).Position);
                if Posi_dim==2
                    plot(posi(1,:),posi(2,:),'r','LineWidth',1);
                elseif Posi_dim==3
                    plot(posi(1,:),posi(2,:),posi(3,:),'r','LineWidth',1);
                end
            end
        end
    end
    scatter(Agents_Posi(1,:),Agents_Posi(2,:),600,'r','.')

    xlabel('x1')
    ylabel('x2')
    colorbar
    xlim([range_x1(1) range_x1(2)])
    ylim([range_x2(1) range_x2(2)])
    %     title('network topology on 2D field')
    hold off
    fname=strcat(results_dir,'/topology_background');
    s=hgexport('factorystyle');
    s.Resolution=300;
    s.FontSizeMin=14;
    s.Format='png';
    s.Width=5;
    hgexport(gcf,fname,s);
    close gcf;

    gcf=figure('visible','on');
    hold on;
    scatter(Agents_Posi(1,:),Agents_Posi(2,:))
    for m=1:M
        for n=m:M
            if A_full(m,n)>0
                posi=[Agents(m).Position,Agents(n).Position];
                Posi_dim=length(Agents(m).Position);
                if Posi_dim==2
                    plot(posi(1,:),posi(2,:),'r','LineWidth',1);
                elseif Posi_dim==3
                    plot(posi(1,:),posi(2,:),posi(3,:),'r','LineWidth',1);
                end
            end
        end
    end
    scatter(Agents_Posi(1,:),Agents_Posi(2,:),600,'r','.')
    for m=1:M
        text(Agents_Posi(1,m),Agents_Posi(2,m),num2str(m));
    end
    xlabel('x1')
    ylabel('x2')
    hold off
    fname=strcat(results_dir,'/just_topology');
    s=hgexport('factorystyle');
    s.Resolution=300;
    s.FontSizeMin=14;
    s.Format='png';
    hgexport(gcf,fname,s);
    close gcf;
end
clear Topology_method G L v posi region Posi_dim X1 X2;
%% Experiment group setup

run_flags =    [run_GD;
    run_ADMM;
    run_pxADMM;
    run_ADMM_fd;
    run_pxADMM_fd_fast_sync;
    run_pxADMM_fd_fast_async;
    run_pxADMM_fd_sync;
    run_pxADMM_fd_async];
% methods_pool stores the name of the methods
methods_pool=["GD";
    "ADMM";
    "pxADMM";
    "ADMM_fd";
    "pxADMM_fd_sync";
    "pxADMM_fd_async";
    "pxADMM_fd_tc_sync";
    "pxADMM_fd_tc_async"];
show_txt=[];
for i=1:length(run_flags)
    if run_flags(i)
        show_txt=[show_txt;methods_pool(i)];
    end
end
disp("The method(s) to be evaluated in this experiment is(are):")
fprintf("%s\n",show_txt);
fprintf("\n");
fprintf("The simulations runs with \rrho=%d, L=%d\n", rho_glb,L_glb);
fprintf("\n");
fprintf("The simulations runs with initial hyperparameter:\r" + ...
    "sigma_f=%.2f,\rl_1=%.2f,\rl_2=%.2f,\rsigma_n=%.4f\n", initial_sigma_f,initial_l(1),initial_l(2),initial_sigma_n);


clear show_txt
%%
% delete(gcp('nocreate'))
% parpool(M)
% rng('shuffle')
%% Perform naive GD
% initial_l=3*[1,1];
% initial_l(1)=initial_l(1)*0.8;
if run_GD
    % run GD
    stepSize=GD_step_size;
    maxIter=15000;

    disp('Time of GD')

    tic
    [sigma_GD,l_GD,sigma_n_GD,Steps_GD,NLLs_GD]  = runGD(Agents,M,initial_sigma_f,initial_l,initial_sigma_n,stepSize,epsilon,maxIter);
    toc

    disp('GD optimization results')
    disp([sigma_GD,l_GD',sigma_n_GD])

    pause(0.1)
end
%% Perform ADMM
if run_ADMM
    % initialize ADMM
    stepSize=GD_step_size; % step size of optimizing theta in inner interations
    initial_beta = [1;ones(length(initial_l),1);1];
    initial_z = [initial_sigma_f;initial_l';initial_sigma_n];
    maxOutIter=2000;
    maxInIter=50;
    for m=1:M
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).mu=stepSize;
    end
    % run ADMM

    disp('Time of ADMM')

    tic
    [sigma_ADMM,l_ADMM,sigma_n_ADMM,Steps_ADMM,IterCounts] = runADMM(Agents,M,stepSize,epsilon,maxOutIter,maxInIter);
    toc

    disp('ADMM optimization results')
    disp([sigma_ADMM,l_ADMM',sigma_n_ADMM])
    %     Steps_ADMM=Steps_ADMM{1};
    pause(0.1)
end
%% Perform pxADMM
if run_pxADMM
    % initialize pxADMM
    maxIter=15000;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = 1*[1;ones(length(initial_l),1);1];
    for m=1:M
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).L=L_glb;
    end
    % run pxADMM

    disp('Time of pxADMM')

    tic
    [sigma_pxADMM,l_pxADMM,sigma_n_pxADMM,Steps_pxADMM,Zs_pxADMM,NLLs_pxADMM] = runPXADMM(Agents,M,epsilon,maxIter);
    toc

    disp('pxADMM optimization results')
    disp([sigma_pxADMM,l_pxADMM',sigma_n_pxADMM])

    pause(0.1)
end
%% Perform ADMM_fd
if run_ADMM_fd
    % initialize ADMM_fd
    maxOutIter=4000;
    maxInIter=50;
    stepSize=GD_step_size;
    for m=1:M
        Agents(m).rho=rho_glb;
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).l=initial_l';
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).mu=stepSize;
        Agents(m).z=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=ones(inputDim+2,Agents(m).N_size);
        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
    end
    for m=1:M
        for n=1:Agents(m).N_size
            neighbor_idx=Agents(m).Neighbors(n);
            Agents(m).theta_n(:,n)=[Agents(neighbor_idx).sigma_f;Agents(neighbor_idx).l;Agents(neighbor_idx).sigma_n];
        end
    end


    % run ADMM_fd

    disp('Time of ADMM_{fd}')

    tic
    [sigma_ADMM_fd,l_ADMM_fd,sigma_n_ADMM_fd,Steps_ADMM_fd,IterCounts_fd] = runADMM_fd(Agents,M,stepSize,epsilon,maxOutIter,maxInIter);
    toc

    disp('ADMM_{fd} optimization results')
    disp([sigma_ADMM_fd,l_ADMM_fd',sigma_n_ADMM_fd])
    %     Steps_ADMM_fd=Steps_ADMM_fd{1};
    pause(0.1)
end
%% Perform pxADMM_fd_sync
if run_pxADMM_fd_fast_sync
    % initialize pxADMM_fd
    maxIter=15000;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = 1*[1;ones(length(initial_l),1);1];
    for m=1:M
        Agents(m).communicationAbility=1;
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).L=L_glb;

        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=ones(inputDim+2,Agents(m).N_size);

        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=zeros(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=2;
    end
    for m=1:M
        for n=1:Agents(m).N_size
            neighbor_idx=Agents(m).Neighbors(n);
            Agents(m).theta_n(:,n)=[Agents(neighbor_idx).sigma_f;Agents(neighbor_idx).l;Agents(neighbor_idx).sigma_n];
        end
    end
    sync=1;
    % run pxADMM_fd

    disp('Time of pxADMM_{fd,sync}')

    tic
    [sigma_pxADMM_fd_sync,l_pxADMM_fd_sync,sigma_n_pxADMM_fd_sync,Steps_pxADMM_fd_sync,Zs_pxADMM_fd,thetas_pxADMM_fd_sync,DataTransNum_pxADMM_fd_sync,NLLs_pxADMM_fd_sync] = ...
        runPXADMM_fd(Agents,M,epsilon,maxIter,sync);
    toc

    disp('pxADMM_{fd,sync} optimization results')
    disp([sigma_pxADMM_fd_sync,l_pxADMM_fd_sync',sigma_n_pxADMM_fd_sync])

    pause(0.1)
    Steps_pxADMM_fd_sync=Steps_pxADMM_fd_sync{1};
end
%% Perform pxADMM_fd_async
if run_pxADMM_fd_fast_async
    % initialize pxADMM_fd
    maxIter=15000;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = 1*[1;ones(length(initial_l),1);1];
    for m=1:M
        Agents(m).communicationAbility=1;

        Agents(m).action_status=1;
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).L=L_glb;

        Agents(m).Zs=initial_z;
        Agents(m).Steps=0;

        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=1*ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=1*ones(inputDim+2,Agents(m).N_size);

        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=ones(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=2;
        Agents(m).updatedVarsNumberThreshold=min(Agents(m).updatedVarsNumberThreshold,Agents(m).N_size);
    end
    for m=1:M
        for n=1:Agents(m).N_size
            neighbor_idx=Agents(m).Neighbors(n);
            Agents(m).theta_n(:,n)=[Agents(neighbor_idx).sigma_f;Agents(neighbor_idx).l;Agents(neighbor_idx).sigma_n];
        end
    end
    sync=0;
    % run pxADMM_fd

    disp('Time of pxADMM_{fd,async}')

    tic
    [sigma_pxADMM_fd_async,l_pxADMM_fd_async,sigma_n_pxADMM_fd_async,Steps_pxADMM_fd_async,Zs_pxADMM_fd_async,thetas_pxADMM_fd_async,DataTransNum_pxADMM_fd_async,NLLs_pxADMM_fd_async] =...
        runPXADMM_fd(Agents,M,epsilon,maxIter,sync);
    toc

    disp('pxADMM_{fd,async} optimization results')
    disp([sigma_pxADMM_fd_async,l_pxADMM_fd_async',sigma_n_pxADMM_fd_async])

    Agents(1).sigma_f=sigma_pxADMM_fd_async;
    Agents(1).l=l_pxADMM_fd_async;
    pause(0.1)
    Steps_pxADMM_fd_async=Steps_pxADMM_fd_async{1};
    NLLs_pxADMM_fd_async=NLLs_pxADMM_fd_async{1};
end

%% run_pxADMM_fd_tc_sync
if run_pxADMM_fd_sync
    % initialize pxADMM_fd_tc_sync
    maxIter=15000;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = 1*[1;ones(length(initial_l),1);1];
    for m=1:M
        Agents(m).communicationAbility=1;
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).L=L_glb;

        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=1*ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=1*ones(inputDim+2,Agents(m).N_size);

        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=zeros(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=2;
    end
    for m=1:M
        for n=1:Agents(m).N_size
            neighbor_idx=Agents(m).Neighbors(n);
            Agents(m).theta_n(:,n)=[Agents(neighbor_idx).sigma_f;Agents(neighbor_idx).l;Agents(neighbor_idx).sigma_n];
        end
    end
    sync=1;
    % run pxADMM_fd

    disp('Time of pxADMM_{fd,tc,sync}')

    tic
    [sigma_pxADMM_fd_tc_sync,l_pxADMM_fd_tc_sync,sigma_n_pxADMM_fd_tc_sync,Steps_pxADMM_fd_tc_sync,Zs_pxADMM_fd_tc,thetas_pxADMM_fd_tc_sync,DataTransNum_pxADMM_fd_tc_sync,NLLs_pxADMM_fd_tc_sync] = ...
        runPXADMM_fd_tc(Agents,M,epsilon,maxIter,sync);
    toc

    disp('pxADMM_{fd,tc,sync} optimization results')
    disp([sigma_pxADMM_fd_tc_sync,l_pxADMM_fd_tc_sync',sigma_n_pxADMM_fd_tc_sync])

    pause(0.1)
    Steps_pxADMM_fd_tc_sync=Steps_pxADMM_fd_tc_sync{1};
end

%% run_pxADMM_fd_tc_async
if run_pxADMM_fd_async
    % initialize pxADMM_fd_tc
    maxIter=15000;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = 1*[1;ones(length(initial_l),1);1];
    for m=1:M
        Agents(m).communicationAbility=1;
        Agents(m).beta=initial_beta;
        Agents(m).z=initial_z;
        Agents(m).rho=rho_glb;
        Agents(m).l=initial_l';
        Agents(m).sigma_f=initial_sigma_f;
        Agents(m).sigma_n=initial_sigma_n;
        Agents(m).L=L_glb;

        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=1*ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=1*ones(inputDim+2,Agents(m).N_size);

        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=zeros(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=2;
        Agents(m).updatedVarsNumberThreshold=min(Agents(m).updatedVarsNumberThreshold,Agents(m).N_size);
    end
    for m=1:M
        for n=1:Agents(m).N_size
            neighbor_idx=Agents(m).Neighbors(n);
            Agents(m).theta_n(:,n)=[Agents(neighbor_idx).sigma_f;Agents(neighbor_idx).l;Agents(neighbor_idx).sigma_n];
        end
    end
    sync=0;
    % run pxADMM_fd

    disp('Time of pxADMM_{fd,tc,async}')

    tic
    [sigma_pxADMM_fd_tc_async,l_pxADMM_fd_tc_async,sigma_n_pxADMM_fd_tc_async,Steps_pxADMM_fd_tc_async,Zs_pxADMM_fd_tc,thetas_pxADMM_fd_tc_async,DataTransNum_pxADMM_fd_tc_async,NLLs_pxADMM_fd_tc_async] =...
        runPXADMM_fd_tc(Agents,M,epsilon,maxIter,sync);
    toc

    disp('pxADMM_{fd,tc,async} optimization results')
    disp([sigma_pxADMM_fd_tc_async,l_pxADMM_fd_tc_async',sigma_n_pxADMM_fd_tc_async])

    pause(0.1)
    Steps_pxADMM_fd_tc_async=Steps_pxADMM_fd_tc_async{1};
    NLLs_pxADMM_fd_tc_async=NLLs_pxADMM_fd_tc_async{1};
end


save(strcat(results_dir,'/forPlot.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% The exp end here, rest parts are plotting %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare convergence speed in terms of iterations
vec_color='krgcmbykrgcmbykrgcmbykrgcmby';
vec_color=[0 0 0; 0 0 1;
    1 0 0; 1 0.1034 0.7241;0 1 0 ;
    0 1 0.7586;0.6207 0.3103 0.2759;1 0.8276 0;0.5127 0.5127 1;
    0 0.3448 0];
color_ind=1;
close all
gcf=figure;
lgd_txt=[];
hold on;
lwd=3;
lwd_thick=3;
lwd_async=0.6;
tiledlayout(1,1,'TileSpacing','compact');
nexttile(1)
if run_GD
    disp('GD steps')
    length(Steps_GD)
    semilogy(Steps_GD,"LineWidth",lwd_thick,'LineStyle','-','Color','k');
    hold on;
    lgd_txt=[lgd_txt;"GD"];
end
if run_ADMM
    semilogy(IterCounts{1}(1:end-1),Steps_ADMM,"LineWidth",lwd_thick,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    %     semilogy(Steps_ADMM,"LineWidth",lwd_thick,'LineStyle','-');
    hold on;
    lgd_txt=[lgd_txt;"ADMM"];
end
if run_pxADMM
    disp('pxADMM steps')
    length(Steps_pxADMM)
    semilogy(Steps_pxADMM,"LineWidth",lwd_thick,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM"];
end
if run_ADMM_fd
    IterCounts_fd{1}(1)=1;
    semilogy(IterCounts_fd{1}(1:end-1),Steps_ADMM_fd,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    %     semilogy(Steps_ADMM_fd,"LineWidth",lwd,'LineStyle','-');
    hold on;
    lgd_txt=[lgd_txt;"ADMM_{fd}"];
end
if run_pxADMM_fd_fast_async
    disp('pxADMM_afd_fast steps')
    length(Steps_pxADMM_fd_async)
    semilogy(Steps_pxADMM_fd_async,"LineWidth",lwd_async,...
        'LineStyle','--','Color',vec_color(color_ind,:),'HandleVisibility','off');
    semilogy(NaN,"LineWidth",lwd_thick,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd,fast}"];
end
if run_pxADMM_fd_async
    disp('pxADMM_afd steps')
    length(Steps_pxADMM_fd_tc_async)
    semilogy(Steps_pxADMM_fd_tc_async,"LineWidth",lwd_async,...
        'LineStyle','--','Color',vec_color(color_ind,:),'HandleVisibility','off');
    semilogy(NaN,"LineWidth",lwd_thick,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd}"];
end
if run_pxADMM_fd_fast_sync
    disp('pxADMM_fd_fast steps')
    length(Steps_pxADMM_fd_sync)
    semilogy(Steps_pxADMM_fd_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,fast}"];
end
if run_pxADMM_fd_sync
    disp('pxADMM_fd steps')
    length(Steps_pxADMM_fd_tc_sync)
    semilogy(Steps_pxADMM_fd_tc_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd}"];
end
set(gca, 'YScale', 'log');
xlabel('iterations');
ylabel('step size');
% title('log plot of steps-iterations');
lgd=legend(lgd_txt,'Location','northoutside','Orientation', 'Horizontal');
lgd.NumColumns=4;


ax1=gca;
ax2=axes('Position',[.5 .5 .3 .3]);
ax1.FontSize = 15;
hold on;
box on;
color_ind=1;
mag_range=400;
if run_GD
    disp('GD steps')
    length(Steps_GD)
    semilogy(Steps_GD,"LineWidth",lwd_thick,'LineStyle','-','Color','k');
end
if run_ADMM
    semilogy(IterCounts{1}(1:end-1),Steps_ADMM,"LineWidth",lwd_thick,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_pxADMM
    disp('pxADMM steps')
    length(Steps_pxADMM)
    semilogy(Steps_pxADMM,"LineWidth",lwd_thick,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_ADMM_fd
    IterCounts_fd{1}(1)=1;
    semilogy(IterCounts_fd{1}(1:end-1),Steps_ADMM_fd,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_pxADMM_fd_fast_async
    disp('pxADMM_afd_fast steps')
    length(Steps_pxADMM_fd_async)
    semilogy(Steps_pxADMM_fd_async,"LineWidth",lwd_async,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_pxADMM_fd_async
    disp('pxADMM_afd steps')
    length(Steps_pxADMM_fd_tc_async)
    semilogy(Steps_pxADMM_fd_tc_async,"LineWidth",lwd_async,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_pxADMM_fd_fast_sync
    disp('pxADMM_fd_fast steps')
    length(Steps_pxADMM_fd_sync)
    semilogy(Steps_pxADMM_fd_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
if run_pxADMM_fd_sync
    disp('pxADMM_fd steps')
    length(Steps_pxADMM_fd_tc_sync)
    semilogy(Steps_pxADMM_fd_tc_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
end
set(gca, 'YScale', 'log');
xlim([0,mag_range])
xinner_lim=xlim;
yinner_lim=ylim;
set(gcf,'CurrentAxes',ax1);
rectangle('Position',[xinner_lim(1) yinner_lim(1) xinner_lim(2) yinner_lim(2)],'Curvature',0)
hold off;
s=hgexport('factorystyle');
s.Resolution=300;
s.Width=10;
s.Height=6;
s.FontSizeMin=17;
fname=strcat(results_dir,'/HOMethodsCompare');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);

savefig(gcf,strcat(fname,'.fig'))

set(gca, 'XScale', 'log');
fname=strcat(fname,'_log');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);

savefig(gcf,strcat(fname,'.fig'))
pause(0.01)

%% Compare convergence speed in terms of iterations NLL
vec_color='krgcmbykrgcmbykrgcmbykrgcmby';
vec_color=[0 0 0; 0 0 1;
    1 0 0; 1 0.1034 0.7241;0 1 0 ;
    0 1 0.7586;0.6207 0.3103 0.2759;1 0.8276 0;0.5127 0.5127 1;
    0 0.3448 0];
color_ind=1;
close all
gcf=figure;
lgd_txt=[];
hold on;
lwd=1.5;
lwd_thick=1.5;
lwd_async=1.5;
tiledlayout(1,1,'TileSpacing','compact');
nexttile(1)
if run_pxADMM
    disp('pxADMM NLLs')
    length(NLLs_pxADMM)
    semilogy(NLLs_pxADMM,"LineWidth",lwd_thick,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM"];
end
if run_pxADMM_fd_fast_async
    disp('pxADMM_afd_fast NLLs')
    length(NLLs_pxADMM_fd_async)
    semilogy(NLLs_pxADMM_fd_async,"LineWidth",lwd_async,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd,fast}"];
end
if run_pxADMM_fd_async
    disp('pxADMM_afd NLLs')
    length(NLLs_pxADMM_fd_tc_async)
    semilogy(NLLs_pxADMM_fd_tc_async,"LineWidth",lwd_async,'LineStyle','--','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd}"];
end
if run_pxADMM_fd_fast_sync
    disp('pxADMM_fd_fast NLLs')
    length(NLLs_pxADMM_fd_sync)
    semilogy(NLLs_pxADMM_fd_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,fast}"];
end
if run_pxADMM_fd_sync
    disp('pxADMM_fd NLLs')
    length(NLLs_pxADMM_fd_tc_sync)
    semilogy(NLLs_pxADMM_fd_tc_sync,"LineWidth",lwd,'LineStyle','-','Color',vec_color(color_ind,:));
    color_ind=color_ind+1;
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd}"];
end
set(gca, 'YScale', 'log');
xlabel('iterations');
ylabel('NLL');
% title('log plot of steps-iterations');
% lgd=legend(lgd_txt,'Location','northoutside','Orientation', 'Horizontal');
% lgd.NumColumns=4;
hold off;
s=hgexport('factorystyle');
s.Resolution=300;
s.Width=9;
s.Height=4;
s.FontSizeMin=17;
fname=strcat(results_dir,'/HOMethodsCompare_NLL');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
savefig(gcf,strcat(fname,'.fig'))

set(gca, 'XScale', 'log');
set(gca, 'FontSize', 15);

fname=strcat(fname,'_log');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
savefig(gcf,strcat(fname,'.fig'))

pause(0.01)


%% Compare convergence speed in terms of iterations pxADMMs

close all
gcf=figure;
lgd_txt=[];
hold on;
tiledlayout(1,1,'TileSpacing','compact');
nexttile(1)
if run_pxADMM
    semilogy(Steps_pxADMM,"LineWidth",lwd_thick,'Color','k');
    hold on;
    lgd_txt=[lgd_txt;"pxADMM"];
end
if run_pxADMM_fd_fast_async
    semilogy(Steps_pxADMM_fd_async,"LineWidth",lwd_async,'LineStyle','--');
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd,fast}"];
end
if run_pxADMM_fd_fast_sync
    semilogy(Steps_pxADMM_fd_sync,"LineWidth",lwd);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,fast}"];
end
if run_pxADMM_fd_async
    semilogy(Steps_pxADMM_fd_tc_async,"LineWidth",lwd_async,'LineStyle','--');
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{afd}"];
end
if run_pxADMM_fd_sync
    semilogy(Steps_pxADMM_fd_tc_sync,"LineWidth",lwd);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd}"];
end

set(gca, 'YScale', 'log');
% set(gca, 'XScale', 'log');
xlabel('iterations');
ylabel('step size');
% title('log plot of steps-iterations');
lgd=legend(lgd_txt,'Location','northoutside','Orientation', 'Horizontal');
lgd.NumColumns=4;
hold off;
s=hgexport('factorystyle');
s.Resolution=300;
s.Width=10;
s.Height=7;
s.FontSizeMin=15;
fname=strcat(results_dir,'/HOMethodsCompare_pxADMMs');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
set(gca, 'XScale', 'log');
fname=strcat(fname,'_log');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
pause(0.01)

%% Compare convergence speed in terms of iterations pxADMMs in terms of transmission number

close all
gcf=figure;
lgd_txt=[];
hold on;
lwd=1.2;
lwd_async=1;
tiledlayout(1,1,'TileSpacing','compact');
nexttile(1)
if run_pxADMM_fd_fast_async
    semilogy(unique(DataTransNum_pxADMM_fd_async{1}),Steps_pxADMM_fd_async,"LineWidth",lwd_async);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,async}^*"];
end
if run_pxADMM_fd_fast_sync
    semilogy(DataTransNum_pxADMM_fd_sync{1},Steps_pxADMM_fd_sync,"LineWidth",lwd);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,sync}^*"];
end
if run_pxADMM_fd_async
    semilogy(unique(DataTransNum_pxADMM_fd_tc_async{1}),Steps_pxADMM_fd_tc_async,"LineWidth",lwd_async);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,async}"];
end
if run_pxADMM_fd_sync
    semilogy(DataTransNum_pxADMM_fd_tc_sync{1},Steps_pxADMM_fd_tc_sync,"LineWidth",lwd);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,sync}"];
end
set(gca, 'YScale', 'log');
% set(gca, 'XScale', 'log');
xlabel('transmissions');
ylabel('step size');
% title('log plot of steps-iterations');
lgd=legend(lgd_txt,'Location','northoutside','Orientation', 'Horizontal');
lgd.NumColumns=4;
hold off;
s=hgexport('factorystyle');
s.Resolution=300;
s.Width=10;
s.Height=7;
s.FontSizeMin=14;
fname=strcat(results_dir,'/HOMethodsCompare_pxADMMs_transnum');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
set(gca, 'XScale', 'log');
fname=strcat(fname,'_log');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
pause(0.01)
%%
save(strcat(results_dir,'/workspaceForDebug.mat'));

disp('Code end at HO')

end_time=datetime('now');
disp(strcat("The code ends at ",datestr(end_time)))
code_duration=end_time-start_time;
disp(strcat("The code duration is ",string(code_duration)))

disp('all code ended')

