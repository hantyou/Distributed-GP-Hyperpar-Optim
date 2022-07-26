%% This is branch main
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
if usejava('desktop')
    figureVisibility='on';
else
    figureVisibility='off';
end
set(0,'DefaultFigureVisible',figureVisibility)

clear figureVisibility F
%% Define field for monitoring
range_x1=[-5,5];
range_x2=[-5,5];
range=[range_x1;range_x2];

rng(990611,'twister')
rand(17+16,1);
M=12;
region=[];
%% parpool setup
delete(gcp('nocreate'))
try
    parpool(32);
catch
    parpool(8)
end

%% Generate/Load dataset
reso_m=256;
reso_n=256;
reso=[reso_m,reso_n];
TotalNumLevel=2500;
everyAgentsSampleNum=floor(TotalNumLevel/M);
Agents_measure_range=4;
realDataSet=0;
if realDataSet==1
    disp('This exp is down with real dataset loaded')
    loadRealDataset
    realz=[0,0,0,0];
else
    disp('This exp is down with artificial dataset loaded')
    temp_data=0;
    realz=[5,1,1,0];
    [F_true,reso]=loadDataset(1,reso,range,realz(1:(end-1)));
    [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));

    %% Decide sample points
    samplingMethod=2; % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
    subSize=ones(M,1)*everyAgentsSampleNum;
    Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,M)*0.9;
        unifrnd(range_x2(1),range_x2(2),1,M)*0.9];
    [X,subSize,sampleIdx] = decideSamplePoints(samplingMethod,subSize,range,Agents_Posi,Agents_measure_range);
    sampleSize=sum(subSize);
    X1=X(1,:);
    X2=X(2,:);
    sigma_n=sqrt(0.1);
    realz(end)=sigma_n;
    sampleError=randn(1,sampleSize)*sigma_n;
    %% Take sample
    Y=interp2(mesh_x1,mesh_x2,F_true,X1,X2);
    agentsPosiY=interp2(mesh_x1,mesh_x2,F_true,Agents_Posi(1,:),Agents_Posi(2,:));
    Z=Y+sampleError; % The observation model (measurement model)
end
%% Dataset division and agents initialization
% Random distribution
% %     localDataSetsSize=sampleSize/M;
% %     % data set division
% %     idx1 = randperm(sampleSize); % generate random index
% %     idx = reshape(idx1,[sampleSize/M,M])'; % divide index into M groups
% %     idxedZ=Z(idx);
% %     subDataSetsZ=reshape(Z(idx),[M,sampleSize/M]); % divide training output
% %     idxedX=X(:,idx1);
% %     subDataSetsX=reshape(idxedX,[2,sampleSize/M,M]); % divide training input
% agents initialization
% Distribute according to range of detection
localDataSetsSize=subSize;
idx1=1:sampleSize;
idx=ones(M,max(subSize));
for m=1:M
    idx(m,1:subSize(m))=idx1(sampleIdx(m)+1:sampleIdx(m+1));
end
idxedZ=Z(idx);
subDataSetsZ=idxedZ;
%     subDataSetsZ=reshape(Z(idx),[M,sampleSize/M]); % divide training output
inputDim=size(X,1);
idxedX=zeros(inputDim,max(subSize),M);
for m=1:M
    idxedX(:,:,m)=X(:,idx(m,:));
end
subDataSetsX=idxedX;
%     subDataSetsX=reshape(idxedX,[2,sampleSize/M,M]); % divide training input


% Generate agents
subSize
Agents=agent.empty(M,0);
for m=1:M
    Agents(m).Code=m;
    Agents(m).TotalNumLevel=TotalNumLevel;
    Agents(m).Z=subDataSetsZ(m,1:subSize(m))';
    Agents(m).X=subDataSetsX(:,1:subSize(m),m);
    Agents(m).idx=idx(m,1:subSize(m));
    Agents(m).N_m=localDataSetsSize(m);
    Agents(m).M=M;
    Agents(m).action_status=1;
    Agents(m).commuRange=4;
    Agents(m).realdataset=realDataSet;
    %     Agents(m).commuRange=2.5;
    Agents(m).realz=realz;
    Agents(m).distX1=dist(Agents(m).X(1,:)).^2;
    Agents(m).distX2=dist(Agents(m).X(2,:)).^2;
    Agents(m).distXd=zeros(subSize(m),subSize(m),inputDim);
    Agents(m).Position=Agents_Posi(:,m);
    for d=1:inputDim
        Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2;
    end
end
top_dir_path='results/HO/';
folder_name=strcat(num2str(M),'_a_',num2str(TotalNumLevel),'_pl');
mkdir(top_dir_path,folder_name);
results_dir=strcat(top_dir_path,'/',folder_name);
clear idx1 idx idexedZ idexedX subDataSetsX
%% Plot field and sampled points and noisy sample points
    gcf=figure;
if realDataSet==1&&temp_data==1
    hold on;
    for i=1:cityNum
        plot(temp_17(:,i));
    end
    hold off;
    xlabel('day')
    ylabel('temperature')
    figure,scatter(coord45(1:cityNum,1),coord45(1:cityNum,2))
elseif realDataSet==0
    surf(mesh_x1,mesh_x2,F_true,'edgecolor','none') % plot the field
    hold on;
    scatter3(X1,X2,Y,'r*')
    scatter3(X1,X2,Z,'b.')
    scatter3(Agents_Posi(1,:),Agents_Posi(2,:),agentsPosiY+1,'k^','filled')
    xlabel('x1')
    ylabel('x2')
    zlabel('environmental scalar value')
    pause(0.01)
end
close gcf
%%
% theta_range=[[0.1,1.1];[-1,log(5)/log(10)]];
% LL=generateLikelihoodMap(X,Z,theta_range,sigma_n);
%% Set topology
Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
A_full=generateTopology(Agents,Topology_method);

for m=1:M
    Agents(m).A=A_full(1:M,1:M);
    Agents(m).Neighbors=find(Agents(m).A(Agents(m).Code,:)~=0);
    Agents(m).N_size=length(Agents(m).Neighbors);
end

G=graph(A_full(1:M,1:M));
% figure,plot(G)
L = laplacian(G);
[~,v]=svd(full(L));
v=diag(v);
if v(end-1)==0
    disp("Error: graph not connected")
end
if realDataSet==0||temp_data==3
    gcf=figure('visible','off');
    hold on;
    if realDataSet==0
        imagesc(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n),F_true);
    end
    if temp_data==3
        imagesc(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(2),range_x2(1),reso_n),F_true);
        %        contour(linspace(range_x1(1),range_x1(2),reso_n),linspace(range_x2(2),range_x2(1),reso_m),F_true,linspace(0,25,10),'-k','LineWidth',0.6);
    end
    scatter(X1,X2,'k*');
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
    %     for m=1:M
    %         text(Agents_Posi(1,m),Agents_Posi(2,m),num2str(m));
    %     end
    %     % colormap("jet")

    xlabel('x1')
    ylabel('x2')
    colorbar
    xlim([range_x1(1) range_x1(2)])
    ylim([range_x2(1) range_x2(2)])
    title('network topology on 2D field')
    hold off
    fname=strcat(results_dir,'/topology_background');
    saveas(gcf,fname,'png');
    close gcf;



    gcf=figure('visible','off');
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
    %     % colormap("jet")
    %     xlim([range_x1(1) range_x1(2)])
    %     ylim([range_x2(1) range_x2(2)])

    xlabel('x1')
    ylabel('x2')
    title('network topology')
    hold off
    fname=strcat(results_dir,'/just_topology');
    saveas(gcf,fname,'png');
    close gcf;
end

clear Topology_method G L v posi;
%% Experiment group setup
run_GD=1;
run_ADMM=1;
run_pxADMM=1;
run_ADMM_fd=1;
run_pxADMM_fd_sync=1;
run_pxADMM_fd_async=1;
run_pxADMM_async_realSimu=0;

run_flags =    [run_GD;
    run_ADMM;
    run_pxADMM;
    run_ADMM_fd;
    run_pxADMM_fd_sync;
    run_pxADMM_fd_async;
    run_pxADMM_async_realSimu];
methods_pool=["GD";
    "ADMM";
    "pxADMM";
    "ADMM_fd";
    "pxADMM_fd_sync";
    "pxADMM_fd_async";
    "pxADMM_async_realSimu"];
show_txt=[];
for i=1:length(run_flags)
    if run_flags(i)
        show_txt=[show_txt;methods_pool(i)];
    end
end
disp("The method(s) to be evaluated in this experiment is(are):")
fprintf("%s\n",show_txt);
fprintf("\n");

% initialize theta and other parameters
if realDataSet==0
    initial_sigma_f=3;
    initial_sigma_n=1;
    initial_l=2*ones(1,inputDim);
else
    initial_sigma_f=5.5;
    initial_sigma_n=0.5;
    initial_l=2*ones(1,inputDim);
end

epsilon = 1e-5; % used for stop criteria
rho_glb=TotalNumLevel*0.2;
L_glb=TotalNumLevel*0.8;

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
    stepSize=0.00001;
    maxIter=8000;

    disp('Time of GD')

    tic
    [sigma_GD,l_GD,Steps_GD]  = runGD(Agents,M,initial_sigma_f,initial_l,initial_sigma_n,stepSize,epsilon,maxIter);
    toc

    pause(0.1)
end
%% Perform ADMM
if run_ADMM
    % initialize ADMM
    stepSize=0.00001; % step size of optimizing theta in inner interations
    initial_beta = [1;ones(length(initial_l),1);1];
    initial_z = [initial_sigma_f;initial_l';initial_sigma_n];
    maxOutIter=1000;
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

    pause(0.1)
end
%% Perform pxADMM
if run_pxADMM
    % initialize pxADMM
    maxIter=10000;
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
    [sigma_pxADMM,l_pxADMM,sigma_n_pxADMM,Steps_pxADMM,Zs_pxADMM] = runPXADMM(Agents,M,epsilon,maxIter);
    toc

    pause(0.1)
end
%% Perform ADMM_fd
if run_ADMM_fd
    % initialize ADMM_fd
    maxOutIter=1000;
    maxInIter=50;
    stepSize=0.00001;
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

    pause(0.1)
end
%% Perform pxADMM_fd_sync
if run_pxADMM_fd_sync
    % initialize pxADMM_fd
    maxIter=8000;
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
        Agents(m).beta_mn=0.1*ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=0.1*ones(inputDim+2,Agents(m).N_size);

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

    disp('Time of pxADMM_{fd}')

    tic
    [sigma_pxADMM_fd_sync,l_pxADMM_fd_sync,sigma_n_pxADMM_fd_sync,Steps_pxADMM_fd_sync,Zs_pxADMM_fd,thetas_pxADMM_fd_sync] = runPXADMM_fd(Agents,M,epsilon,maxIter,sync);
    toc

    pause(0.1)
    Steps_pxADMM_fd_sync=Steps_pxADMM_fd_sync{1};
end
%% Perform pxADMM_fd_async
if run_pxADMM_fd_async
    % initialize pxADMM_fd
    maxIter=10000;
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
        Agents(m).beta_mn=0.1*ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=0.1*ones(inputDim+2,Agents(m).N_size);

        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=ones(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=3;
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

    disp('Time of pxADMM_{fd}')

    tic
    [sigma_pxADMM_fd_async,l_pxADMM_fd_async,sigma_n_pxADMM_fd_async,Steps_pxADMM_fd_async,Zs_pxADMM_fd_async,thetas_pxADMM_fd_async] =...
        runPXADMM_fd(Agents,M,epsilon,maxIter,sync);
    toc
    Agents(1).sigma_f=sigma_pxADMM_fd_async;
    Agents(1).l=l_pxADMM_fd_async;
    pause(0.1)
    Steps_pxADMM_fd_async=Steps_pxADMM_fd_async{1};
end
%% Perform pxADMM_async_realSimu
if run_pxADMM_async_realSimu
delete(gcp('nocreate'))
parpool(M)
    maxIter=300;
    initial_z=[initial_sigma_f;initial_l';initial_sigma_n];
    initial_beta = [1;ones(length(initial_l),1);1];
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
        Agents(m).maxIter=maxIter;
        Agents(m).Zs=initial_z;
        Agents(m).Steps=0;
        Agents(m).M=M;
        Agents(m).z_mn=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).z_nm=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_mn=ones(inputDim+2,Agents(m).N_size);
        Agents(m).beta_nm=ones(inputDim+2,Agents(m).N_size);
        Agents(m).currentStep=1;
        Agents(m).theta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).beta_n=zeros(inputDim+2,Agents(m).N_size);
        Agents(m).updatedVars=ones(1,Agents(m).N_size);
        Agents(m).updatedVarsNumberThreshold=2;
        Agents(m).updatedVarsNumberThreshold=min(Agents(m).updatedVarsNumberThreshold,Agents(m).N_size);
        Agents(m).slowSync=zeros(M,1);
        Agents(m).slowSyncThreshold=2;
        Agents(m).runningHO=1;
    end

    disp('Time of pxADMM_{fd,realSimu}')
    [Agents,cinfo] = runPXADMM_fd_spmd(Agents,epsilon);

    figure('visible','off');
    for m=1:M
        subplot(M,1,m);
        imagesc(cinfo{m});
        colorbar;
    end


    gcf=figure;
    hold on
    for m=1:M
        plot(Agents(m).Steps(2:end));
    end
    hold off
    set(gca,'YScale','log')
    fname=strcat(results_dir,'/pxADMM_fd_spmd_steps');
    saveas(gcf,fname,'png');
    close gcf

    gcf=figure;
    for i=1:length(initial_z)
        if i==1
            disp('sigma_f')
        elseif i==inputDim+2
            disp('sigma_n')
        else
            disp(strcat('l_',num2str(i-1)))
        end
        subplot(length(initial_z),1,i)
        hold on
        for m=1:M
            plot(Agents(m).Zs(i,2:end));
%             disp(Agents(m).Zs(i,end))
        end
        hold off
        %set(gca,'YScale','log')
        set(gca,'XScale','log')
    end
    fname=strcat(results_dir,'/pxADMM_fd_spmd_vars');
    saveas(gcf,fname,'png');
    close gcf

    warning('on','all')
end
%% Compare convergence speed in terms of iterations
gcf=figure;
lgd_txt=[];
hold on;
if run_GD
    semilogy(Steps_GD);
    hold on;
    lgd_txt=[lgd_txt;"GD"];
end
if run_ADMM
    semilogy(IterCounts{1}(1:end-1),Steps_ADMM);
    hold on;
    lgd_txt=[lgd_txt;"ADMM"];
end
if run_pxADMM
    semilogy(Steps_pxADMM);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM"];
end
if run_ADMM_fd
    IterCounts_fd{1}(1)=1;
    semilogy(IterCounts_fd{1}(1:end-1),Steps_ADMM_fd);
    hold on;
    lgd_txt=[lgd_txt;"ADMM_{fd}"];
end
if run_pxADMM_fd_sync
    semilogy(Steps_pxADMM_fd_sync);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,sync}"];
end
if run_pxADMM_fd_async
    semilogy(Steps_pxADMM_fd_async);
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{fd,async}"];
end
if run_pxADMM_async_realSimu
    semilogy(Agents(1).Steps(2:end));
    hold on;
    lgd_txt=[lgd_txt;"pxADMM_{async,realSimu}"];
end
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
xlabel('iterations');
ylabel('norm(step)');
title('log plot of steps-iterations');
lgd=legend(lgd_txt,'Location','northoutside','Orientation', 'Horizontal');
lgd.NumColumns=4;
hold off;
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname=strcat(results_dir,'/HOMethodsCompare');
hgexport(gcf,fname,s);
pause(0.01)
%%
save(strcat(results_dir,'/workspaceForDebug.mat'));

disp('Code end at HO')

end_time=datetime('now');
disp(strcat("The code ends at ",datestr(end_time)))
code_duration=end_time-start_time;
disp(strcat("The code duration is ",string(code_duration)))
return
%% GPR real
realDataSet=0;
range_x1=[min(X(1,:)),max(X(1,:))];
range_x2=[min(X(2,:)),max(X(2,:))];
if 0
    sigma_f=Agents(1).sigma_f;
    l=Agents(1).l;
    vacantdata=[];
    X_train=X;
    X_train(:,vacantdata)=[];
    Z_train=Z;
    Z_train(:,vacantdata)=[];
    newXs=X;
    [predictY_mean,predictY_var] = GPR_predictReal(X_train,Z_train,[sigma_f;l],newXs);
    %%
    newDayLength=max(newXs(3,:))-min(newXs(3,:))+1;
    figure,
    col=5;
    col=ceil(col);
    for c=1:cityNum
        subplot(ceil(cityNum/col),col,c)
        hold on
        plot(temp_17_train(:,c));
        plot(1+linspace(1,newDayLength,size(predictY_mean,2)),predictY_mean(c,:));
        hold off

    end

    figure,
    for c=1:cityNum
        subplot(ceil(cityNum/col),col,c)
        hold on
        plot(predictY_var(c,:));
        hold off

    end
else

    %% GPR1

     delete(gcp('nocreate'))
 
     try
         parpool(24);
     catch
         parpool(8)
     end

    % theta=[Agents(1).sigma_f;Agents(1).l];
    theta=[sigma_pxADMM_fd_sync;l_pxADMM_fd_sync];
    theta
    for m=1:M
        Agents(m).sigma_f=thetas_pxADMM_fd_sync(1,m);
        Agents(m).l=thetas_pxADMM_fd_sync(2:end-1,m);
        Agents(m).sigma_n=thetas_pxADMM_fd_sync(end,m);

        [Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]
    end
    %% Pre
    reso_x=100;
    reso_y=100;
    sigma_n
    ts_1=linspace(range_x1(1),range_x1(2),reso_x);
    ts_2=linspace(range_x2(1),range_x2(2),reso_y);
    [mesh_x,mesh_y]=meshgrid(ts_1,ts_2);

    vecX=mesh_x(:);

    vecY=mesh_y(:);
    newX=[vecX,vecY]';

    plotFlag=1;
    fig_export_pix=300;
    eps_export=0;
    png_export=1;
    contourFlag=0;

    %%
    method='PoE';
    [MeanPoE,VarPoE] = GPR_predict_central(Agents,method,newX,sigma_n);
    MeanPoE=reshape(MeanPoE,reso_x,reso_y);
    VarPoE=reshape(VarPoE,reso_x,reso_y);


    Mean=MeanPoE;
    Var=VarPoE;

    gcf=figure('visible','off');
    tiledlayout(2,4,'TileSpacing','Compact','Padding','Compact');

    ax1=nexttile(1);

    surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    for m=1:M
        % scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Agents(m).Z,'*')
    end
    scatter3(Agents_Posi(1,:),Agents_Posi(2,:),agentsPosiY+1,'k^','filled')
    hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title(strcat(method,' GPR result - mean'));
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    caxis(ax1,[6,18]);

    view(0,90);
    %     subplot(245),
    ax5=nexttile(5)

    surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
    for m=1:M
        Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
        scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
    end
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1'), ylabel('x2'), zlabel('y')
    zlim(10.^[-4.1,2.9])
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    zticks(10.^(-4:2:2));
    title({strcat(method,' GPR result'),'variance (in log plot)'})

    %view(0,90);
    %% gPoE
    method='gPoE';


    [MeangPoE,VargPoE] = GPR_predict_central(Agents,method,newX,sigma_n);
    MeangPoE=reshape(MeangPoE,reso_x,reso_y);
    VargPoE=reshape(VargPoE,reso_x,reso_y);


    Mean=MeangPoE;
    Var=VargPoE;

    ax2=nexttile(2);

    surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    for m=1:M
        %scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Agents(m).Z,'*')
    end
    scatter3(Agents_Posi(1,:),Agents_Posi(2,:),agentsPosiY+1,'k^','filled')
    hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title(strcat(method,' GPR result - mean'));
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    caxis(ax2,[6,18])
    %     subplot(246),
    view(0,90);
    nexttile(6)

    surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
    for m=1:M
        Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
        scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
    end
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1'), ylabel('x2'), zlabel('y')
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    zlim(10.^[-4.1,2.9])
    zticks(10.^(-4:2:2));
    title({strcat(method,' GPR result'),'variance (in log plot)'})
    %view(0,90);

    %% BCM
    method='BCM';


    [MeanBCM,VarBCM] = GPR_predict_central(Agents,method,newX,sigma_n);
    MeanBCM=reshape(MeanBCM,reso_x,reso_y);
    VarBCM=reshape(VarBCM,reso_x,reso_y);


    Mean=MeanBCM;
    Var=VarBCM;

    %     figure,subplot(121)
    %     subplot(243)
    ax3=nexttile(3);
    surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    for m=1:M
        %scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Agents(m).Z,'*')
    end
    scatter3(Agents_Posi(1,:),Agents_Posi(2,:),agentsPosiY+1,'k^','filled')
    hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title(strcat(method,' GPR result - mean'));
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    caxis(ax3,[6,18])

    %     subplot(247),
    view(0,90);
    nexttile(7)

    surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
    for m=1:M
        Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
        scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
    end
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1'), ylabel('x2'), zlabel('y')
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    zlim(10.^[-4.1,2.9])
    zticks(10.^(-4:2:2));
    title({strcat(method,' GPR result'),'variance (in log plot)'})
    % view(0,90);


    %% rBCM
    method='rBCM';


    [MeanrBCM,VarrBCM] = GPR_predict_central(Agents,method,newX,sigma_n);
    MeanrBCM=reshape(MeanrBCM,reso_x,reso_y);
    VarrBCM=reshape(VarrBCM,reso_x,reso_y);


    Mean=MeanrBCM;
    Var=VarrBCM;

    %     subplot(244)
    ax4=nexttile(4);

    surf(mesh_x,mesh_y,(Mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    for m=1:M
        % scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Agents(m).Z,'*')
    end
    scatter3(Agents_Posi(1,:),Agents_Posi(2,:),agentsPosiY+1,'k^','filled')
    hold off; xlabel('x1'), ylabel('x2'), zlabel('y'), title(strcat(method,' GPR result - mean'));
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    caxis(ax4,[6,18])

    %     subplot(248),
    view(0,90);
    nexttile(8)

    surf(mesh_x,mesh_y,(Var)/1,'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    Z_ps=interp2(mesh_x,mesh_y,(Var),X(1,:),X(2,:));
    for m=1:M
        Z_ps=interp2(mesh_x,mesh_y,(Var)/1,Agents(m).X(1,:),Agents(m).X(2,:));
        scatter3(Agents(m).X(1,:),Agents(m).X(2,:),Z_ps,'*');
    end
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1'), ylabel('x2'), zlabel('y')
    xlim([range_x1(1),range_x1(2)]);
    ylim([range_x2(1),range_x2(2)]);
    zlim(10.^[-4.1,2.9])
    zticks(10.^(-4:2:2));
    title({strcat(method,' GPR result'),'variance (in log plot)'})
    %  view(0,90);



    sname='CEN predict plot';
    s=hgexport('readstyle',sname);
    s.Resolution=fig_export_pix;
    if eps_export==1
        s.Format='eps';
        fname=strcat('./results/Agg/Centralized/','centralized-GPR-predict');
        if temp_data==3
            fname=strcat(fname,'_',region);
        end
        fname=strcat(fname,'.',s.Format);
        hgexport(gcf,fname,s);
        disp("eps file saved")
    end
    if png_export==1
        s.Format='png';
        fname=strcat('./results/Agg/Centralized/','centralized-GPR-predict.');
        if temp_data==3
            fname=strcat(fname,'_',region);
        end
        fname=strcat(fname,'.',s.Format);
        hgexport(gcf,fname,s);
        disp("png file saved")
    end


    %     pause(1)
    fname=strcat('./results/Agg/Centralized/','centralized-GPR-predict');

    saveas(gcf,fname,'fig');
    saveas(gcf,strcat(fname,'_direct_save'),'png');
    close(gcf)
    %% GPR Full
    method='Full';
    disp('Full')

    tic
    [Mean_total,Uncertainty_total] = GPR_predict(X,Z,theta,[range_x1;range_x2],sigma_n,plotFlag);
    toc

    Mean_total=reshape(Mean_total,1,reso_x*reso_y);
    Uncertainty_total=reshape(Uncertainty_total,1,reso_x*reso_y);
    if temp_data==2
        ts_1=linspace(range_X1(1),range_X1(2),100);
        ts_2=linspace(range_X2(1),range_X2(2),100);
        [mesh_X1,mesh_X2]=meshgrid(ts_1,ts_2);
        GT = shaperead("Road_LAeq_16h_London\Road_LAeq_16h_London.shp");
        figure,
        mapshow(GT);
        hold on;
        surf(mesh_X1,mesh_X2,(Mean_total),'edgecolor','none','FaceAlpha',0.9);
        hold off
    end

    %% No aggregation
    method='NoAg';

    tic
    [meanNoAg,varNoAg] = GPR_predict_NoAg(Agents,newX,sigma_n);
    toc
    %%
    visible='off';
    fname=strcat('./results/Agg/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanNoAg,varNoAg,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);


    %% DEC-PoE
    method='DEC-PoE';

    A=A_full(1:M,1:M);
    maxIter=20;

    tic
    [~,~,meanDEC_PoE,varDEC_PoE] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
    toc
    visible='off';
    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanDEC_PoE,varDEC_PoE,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);



    %% DEC-gPoE
    method='DEC-gPoE';

    A=A_full(1:M,1:M);
    maxIter=30;

    tic
    [~,~,meanDEC_gPoE,varDEC_gPoE] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
    toc
    visible='off';
    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanDEC_gPoE,varDEC_gPoE,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);


    %% DEC-BCM
    method='DEC-BCM';

    A=A_full(1:M,1:M);
    maxIter=30;

    tic
    [~,~,meanDEC_BCM,varDEC_BCM] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
    toc
    visible='off';
    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanDEC_BCM,varDEC_BCM,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);


    %% DEC-rBCM
    method='DEC-rBCM';

    A=A_full(1:M,1:M);
    maxIter=30;

    tic
    [~,~,meanDEC_rBCM,varDEC_rBCM] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
    toc
    visible='off';
    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanDEC_rBCM,varDEC_rBCM,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);


    %% DEC-NPAE
    method='DEC-NPAE';

    A=A_full(1:M,1:M);
    maxIter=30;

    tic
    [~,~,meanDEC_NPAE,varDEC_NPAE] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
    toc
    visible='off';
    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanDEC_NPAE,varDEC_NPAE,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);
    %% NN-NPAE
    method='NN-NPAE';
    tic
    [meanNN_NPAE,varNN_NPAE] = GPR_predict_NN(Agents,method,newX,sigma_n);
    toc

    visible='off';

    fname=strcat('./results/Agg/DEC/',strcat(method,'-GPR-predict'));
    agentsPredictionPlot(Agents,meanNN_NPAE,varNN_NPAE,reso_x,reso_y,...
        range_x1,range_x2,agentsPosiY,fname,method,eps_export,png_export,...
        visible,fig_export_pix,temp_data,region,contourFlag);

    %% Evaluate Prediction Performance
    evaMethod='RMSE';
    disp('evaluating')
    realMean=Mean_total;
    realVar=Uncertainty_total;
    %
    [pfmcMean_NoAg,pfmcVar_NoAg] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanNoAg,varNoAg,evaMethod)
    %
    [pfmcMean_PoE,pfmcVar_PoE] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanDEC_PoE,varDEC_PoE,evaMethod)
    %
    [pfmcMean_gPoE,pfmcVar_gPoE] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanDEC_gPoE,varDEC_gPoE,evaMethod)
    %
    [pfmcMean_BCM,pfmcVar_BCM] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanDEC_BCM,varDEC_BCM,evaMethod)
    %
    [pfmcMean_rBCM,pfmcVar_rBCM] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanDEC_rBCM,varDEC_rBCM,evaMethod)
    %
    [pfmcMean_DEC_NPAE,pfmcVar_DEC_NPAE] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanDEC_NPAE,varDEC_NPAE,evaMethod)
    %
    [pfmcMean_NN_NPAE,pfmcVar_NN_NPAE] = ...
        evaluatePredictionPerformanceMetrices(realMean,realVar,meanNN_NPAE,varNN_NPAE,evaMethod)


    save('workspaceForDebugAfterPrediction.mat');
end
disp('all code ended')

