clc,clear;
close all;
set(0,'DefaultFigureVisible','off')
delete(gcp('nocreate'))

try
    parpool(30);
catch
    parpool(8)
end
%% Data loading
dataSourceOption=2;
%1: Inherit from HPerpar Optimz part with files name as "workspaceForDebug.mat"
%2: Freshly generated data

if dataSourceOption==1
    try
        load('workspaceForDebug.mat')
    catch
        disp("data is not ready")
        return;
    end
    try
        samplingMethod;
    catch
        samplingMethod=method;
    end
    % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
    if samplingMethod==1
        disp("Sampling method: uniformly distirbuted accross region;")
    elseif samplingMethod==2
        disp("Sampling method: near agents position, could lose some points if out of range;")
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
    maxM=16;
    rng(990611,'twister')
    rand(3*maxM,1);
    reso_m=256;
    reso_n=256;
    reso=[reso_m,reso_n];
    everyAgentsSampleNum=100;
    Agents_measure_range=3;
    realDataSet=0;
    samplingMethod=2; % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
    agentsScatterMethod=2; % 1. Randomly distributed accross the area; 2. K_means center
    overlap=1; % 1. overlap allowed (fuzzy c-means), 2. disjoint clusters
    %% Generate/Load dataset
    
    if realDataSet==1
        disp('This exp is down with real dataset loaded')
        loadRealDataset
    else
        disp('This exp is down with artificial dataset loaded')
        temp_data=0;
        [F_true,reso]=loadDataset(1,reso,range,[5,1,1]);
        [mesh_x1,mesh_x2]=meshgrid(linspace(range_x1(1),range_x1(2),reso_m),linspace(range_x2(1),range_x2(2),reso_n));
        
        %% Decide sample points
        
        % renew twister
        rng(990611,'twister')
        
        tic
        disp('decide agents positions');
        if agentsScatterMethod==1
            Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,maxM)*0.9;
                unifrnd(range_x2(1),range_x2(2),1,maxM)*0.9];
        else
            disp('also deciding sampling points');
            Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,maxM)*0.9;
                unifrnd(range_x2(1),range_x2(2),1,maxM)*0.9];
            subSize=ones(maxM,1)*everyAgentsSampleNum;
            [X_temp,subSize,sampleIdx] = decideSamplePoints(1,subSize,range,Agents_Posi,Agents_measure_range);
            
            X_temp=X_temp';
            cVec = 'bgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmy';
            pVec='.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p';
            
            numC=maxM;
            if overlap==1
                fcm_option=[3,50,1e-6,false];
                [centers,U] = fcm(X_temp,numC,fcm_option);
                fcm_idx=zeros(size(U,2),1);
                for n=1:size(U,2)
                    U_n=U(:,n);
                    ind1=find(U_n==max(U_n));
                    p_ind1=U_n(ind1);
                    U_n(U_n==max(U_n))=0;
                    ind2=find(U_n==max(U_n));
                    p_ind2=U_n(ind2);
                    p_ind1=p_ind1/(p_ind1+p_ind2);
                    if rand>p_ind1
                        ind=ind2;
                    else
                        ind=ind1;
                    end
                    
                    fcm_idx(n)=ind;
                end
                idx=fcm_idx;C=centers;
                clear p_ind1 ind1 ind2
            else
                opts = statset('Display','final');
                [idx,C] = kmeans(X_temp,numC,'Distance','cityblock',...
                    'Replicates',5,'Options',opts);
            end
            Agents_Posi=C';
            clusterIdx=idx;
            
            figure;
            hold on
            LegendTxt=cell(numC+1,1);
            for id=1:numC
                plot(X_temp(idx==id,1),X_temp(idx==id,2),[cVec(id) pVec(id)],'MarkerSize',10)
                LegendTxt{id}=strcat('Cluster',num2str(id));
            end
            title 'Cluster Assignments and Centroids'
            LegendTxt{end}='Centroids';
            plot(C(:,1),C(:,2),'kx',...
                'MarkerSize',15,'LineWidth',3)
            legend(LegendTxt,'Location','NW')
            hold off
        end
        toc
        
        
        subSize=ones(maxM,1)*everyAgentsSampleNum;
        
        [X,subSize,sampleIdx] = decideSamplePoints(samplingMethod,subSize,range,Agents_Posi,Agents_measure_range);
        
        if agentsScatterMethod==2
            X=X_temp';
            subSize=zeros(maxM,1);
            for m=1:maxM
                subSize(m)=length(idx(idx==m));
                sampleIdx=cumsum(subSize)';
                sampleIdx=[0,sampleIdx];
            end
            
        end
        
        
        sampleSize=sum(subSize);
        X1=X(1,:);
        X2=X(2,:);
        sigma_n=sqrt(0.1);
        sampleError=randn(1,sampleSize)*sigma_n;
        %% Take sample
        Y=interp2(mesh_x1,mesh_x2,F_true,X1,X2);
        agentsPosiY=interp2(mesh_x1,mesh_x2,F_true,Agents_Posi(1,:),Agents_Posi(2,:));
        Z=Y+sampleError; % The observation model (measurement model)
    end
    %% Dataset division and agents initialization
    % agents initialization
    % Distribute according to range of detection
    localDataSetsSize=subSize;
    idx1=1:sampleSize;
    idx=ones(maxM,max(subSize));
    %     clusterIdx=clusterIdx';
    for m=1:maxM
        if agentsScatterMethod==2
            idx(m,1:subSize(m))=find(clusterIdx==m)';
        else
            idx(m,1:subSize(m))=idx1(sampleIdx(m)+1:sampleIdx(m+1));
        end
    end
    idxedZ=Z(idx);
    subDataSetsZ=idxedZ;
    inputDim=size(X,1);
    idxedX=zeros(inputDim,max(subSize),maxM);
    for m=1:maxM
        idxedX(:,:,m)=X(:,idx(m,:));
    end
    subDataSetsX=idxedX;
    % Generate agents
    subSize;
    Agents=agent.empty(maxM,0);
    for m=1:maxM
        Agents(m).Code=m;
        Agents(m).Z=subDataSetsZ(m,1:subSize(m))';
        Agents(m).X=subDataSetsX(:,1:subSize(m),m);
        Agents(m).idx=idx(m,1:subSize(m));
        Agents(m).N_m=localDataSetsSize(m);
        Agents(m).M=maxM;
        Agents(m).action_status=1;
        Agents(m).commuRange=3.5;
        %     Agents(m).commuRange=2.5;
        
        Agents(m).distX1=dist(Agents(m).X(1,:)).^2;
        Agents(m).distX2=dist(Agents(m).X(2,:)).^2;
        Agents(m).distXd=zeros(subSize(m),subSize(m),inputDim);
        Agents(m).Position=Agents_Posi(:,m);
        for d=1:inputDim
            Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2;
        end
    end
    
    %% Plot field and sampled points and noisy sample points
    
    if realDataSet==1&&temp_data==1
        figure,
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
    
    
    
    %% Set topology
    Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
    A_full=generateTopology(Agents,Topology_method);
    clear Topology_method;
    
    for m=1:maxM
        Agents(m).A=A_full(1:maxM,1:maxM);
        Agents(m).Neighbors=find(Agents(m).A(Agents(m).Code,:)~=0);
        Agents(m).N_size=length(Agents(m).Neighbors);
    end
    
    G=graph(A_full(1:maxM,1:maxM));
    % figure,plot(G)
    L = laplacian(G);
    [~,v]=svd(full(L));
    v=diag(v);
    if v(end-1)==0
        disp("Error: graph not connected")
    end
    clear L;
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
        for m=1:maxM
            for n=m:maxM
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
        fname='results/Agg/PerformanceEva/topology_background';
        saveas(gcf,fname,'png');
        close gcf;
        
        
        
        gcf=figure('visible','on');
        hold on;
        scatter(Agents_Posi(1,:),Agents_Posi(2,:))
        for m=1:maxM
            for n=m:maxM
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
        for m=1:maxM
            text(Agents_Posi(1,m),Agents_Posi(2,m),num2str(m));
        end
        %     % colormap("jet")
        %     xlim([range_x1(1) range_x1(2)])
        %     ylim([range_x2(1) range_x2(2)])
        
        xlabel('x1')
        ylabel('x2')
        title('network topology')
        hold off
        fname='results/Agg/PerformanceEva/just_topology';
        saveas(gcf,fname,'png');
        close gcf;
    end
    
    
    
end
disp('data loading/generation end')
disp('%%%%%%%%%%%%%%%%%%%%Examine Part Begin%%%%%%%%%%%%%%%%%%%%%%%')
close all
clearvars -except A_full Agents Agents_Posi cVec pVec X Y Z range sampleSize sigma_n
%% Evaluation setup
Ms=[2,4,8,12,16]; % different number of agents for different exp groups
tempFlag=[1,1,1,1,1];

maxRange=max(range(:))-min(range(:));
commuRange=[maxRange,maxRange/2,maxRange/3,maxRange/3,maxRange/7];
tempFlag=tempFlag==1;
Ms=Ms(tempFlag);
commuRange=commuRange(tempFlag);

Num_expGroup=length(Ms);
MethodsName=[
    "PoE";
    "BCM";
    "gPoE";
    "rBCM";
    "DEC-PoE";
    "DEC-gPoE";
    "DEC-BCM";
    "DEC-rBCM";
    "DEC-NPAE";
    "NN-NPAE";
    "CON-NPAE";
    "No-Ag"];
DECNAME={
    "DEC-PoE";
    "DEC-gPoE";
    "DEC-BCM";
    "DEC-rBCM";
    "DEC-NPAE"}';
NNNAME={"NN-NPAE"};



NOAGNAME={"No-Ag"};

CENTER={"PoE";
    "BCM";
    "gPoE";
    "rBCM"}';



MethodsFlag=[
    0;
    0;
    0;
    0;
    1;
    1;
    1;
    1;
    1;
    1;
    1;
    1];
MethodsExamined=MethodsName(MethodsFlag==1);
Num_MethodsExamined=sum(MethodsFlag);
disp("Methods to be examined are:")
for n=1:Num_MethodsExamined
    disp(MethodsExamined(n))
end

%% Pre-parpool reset
% delete(gcp('nocreate'))
%
% try
%     parpool(24);
% catch
%     parpool(8)
% end
%% Pre-exp pars
reso_x=100;
reso_y=100;
range_x1=range(1,:);
range_x2=range(2,:);

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

overlapFlag=1; %0. k-means, 1. fcm
theta=[5,1,1]';
Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
savePlot=0;

meanRMSE=zeros(Num_MethodsExamined,Num_expGroup);
varRMSE=zeros(Num_MethodsExamined,Num_expGroup);
graphs=cell(Num_expGroup,1);
times=zeros(Num_MethodsExamined,Num_expGroup);

%% Evaluate Full GP
method='Full';
disp('Now calculating Full GP')

tic
[Mean_total,Uncertainty_total] = GPR_predict(X,Z,theta,[range_x1;range_x2],sigma_n,plotFlag);
toc
realMean=Mean_total;
realMean=reshape(realMean,1,reso_x*reso_y);
realVar=Uncertainty_total;
realVar=reshape(realVar,1,reso_x*reso_y);

%% Evaluate Others

rng(990611,'twister')
for expId=1:Num_expGroup
    M=Ms(expId);
    disp("Agent number:")
    disp(M)
    connected=0;
    
    [Agents,clusterIdx] = predictionEvaluationLoopDataDivide(range,X,Z,M,overlapFlag);
    while connected==0
        for m=1:M
            Agents(m).commuRange=commuRange(expId);
        end
        A_full=generateTopology(Agents,Topology_method);
        G=graph(A_full);
        Lap = laplacian(G);
        [V,~]=eig(full(Lap));
        V=diag(V);
        if V(2)>1e-5
            connected=1;
        else
            commuRange(expId)=commuRange(expId)*1.2;
            continue;
        end
        for m=1:M
            Agents(m).A=A_full(1:M,1:M);
            Agents(m).Neighbors=find(Agents(m).A(Agents(m).Code,:)~=0);
            Agents(m).N_size=length(Agents(m).Neighbors);
            Agents(m).sigma_f=theta(1);
            Agents(m).l=theta(2:end);
            Agents(m).z=theta;
            Agents(m).sigma_n=sigma_n;
        end
    end
    graphs{expId}=G;
    pause(0.01)
    
    for n=1:Num_MethodsExamined
        method=MethodsExamined(n);
        tic
        switch method
            case DECNAME
                disp(method)
                A=A_full(1:M,1:M);
                maxIter=50;
                [~,~,mean,var] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n);
            case NNNAME
                disp(method)
                [mean,var] = GPR_predict_NN(Agents,method,newX,sigma_n);
                meanNN=mean;
                varNN=var;
            case NOAGNAME
                disp(method)
                [mean,var] = GPR_predict_NoAg(Agents,newX,sigma_n);
            case "CON-NPAE"
                disp(method)
                method1="NN-NPAE";
                method2="DEC-BCM";
                NNFlag=0;
                try
                    meanNN;
                    varNN;
                    NNFlag=1;
                catch
                    NNFlag=0;
                end
                
                if NNFlag==0
                    [mean,var] = GPR_predict_NN(Agents,method1,newX,sigma_n);
                    [~,~,mean,var] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,mean,var);
                elseif NNFlag==1
                    [~,~,mean,var] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,meanNN,varNN);
                end
                
            case CENTER
                disp(method)
                disp("!!!!!!!!!!!!!!!!!Evaluation Under Construction!!!!!!!!!!!!!!!!!!!!")
        end
        toc
        TOC=toc;
        times(n,expId)=TOC;
        evaMethod='RMSE';
        [meanRMSE(n,expId),varRMSE(n,expId)] = ...
            evaluatePredictionPerformanceMetrices(realMean,realVar,mean,var,evaMethod);
    end
    
end

save('evalueResult.mat');
%% plot result

gcf=figure;
for m=1:Num_expGroup
    subplot(1,Num_expGroup,m);
    plot(graphs{m});
end
fname='results/Agg/PerformanceEva/Graphs';
saveas(gcf,fname,'png');
close gcf;

gcf=figure;
hold on;
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,meanRMSE(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','NW');
hold off
title('Prediction Mean RMSE')
fname='results/Agg/PerformanceEva/MeanRMSE';
saveas(gcf,fname,'png');
close gcf;

gcf=figure;
hold on;
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,VarRMSE(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','NW');
hold off
title('Prediction Var RMSE')
fname='results/Agg/PerformanceEva/VarRMSE';
saveas(gcf,fname,'png');
close gcf;

gcf=figure;
hold on
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,times(m,:),'-o')
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','NW');
hold off
title('Time used')
fname='results/Agg/PerformanceEva/Times';
saveas(gcf,fname,'png');
close gcf;






