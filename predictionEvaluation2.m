clc,clear;
close all;
start_time=datetime('now');
disp(strcat("The code begins at ",datestr(start_time)))
set(0,'DefaultFigureVisible','off')
%%
delete(gcp('nocreate'))
try
    parpool(8);
catch
    parpool(8)
end
%%
range_x1=[-5,5];
range_x2=[-5,5];
range=[range_x1;range_x2];
maxM=8;
reso_m=256;
reso_n=256;
everyAgentsSampleNum=100;
Agents_measure_range=4;
realDataSet=0;
samplingMethod=1; % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
agentsScatterMethod=2; % 1. Randomly distributed accross the area; 2. K_means center
overlap=1; % 1. overlap allowed (fuzzy c-means), 2. disjoint clusters
reso=[reso_m,reso_n];  
%% Evaluation setup
    Ms=[maxM]; % different number of agents for different exp groups
    sampleNums=[30,60,120,240,300];
    tempFlag=[1];
    tempFlag_outer=[1,1,1,1,1];
    point_number_group=sum(tempFlag_outer);  

    maxRange=max(range(:))-min(range(:));
    commuRange=[maxRange,maxRange/4,maxRange/5,maxRange/6,maxRange/10];
    tempFlag=tempFlag==1;
    tempFlag_outer=tempFlag_outer==1;
    Ms=Ms(tempFlag);
    sampleNums=sampleNums(tempFlag_outer);
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
    
    IANAME=[
        "DEC-PoE";
        "DEC-gPoE";
        "DEC-BCM";
        "DEC-rBCM"];
    
    
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
        0;
        1;
        1;
        1;
        0;
        0;
        0;
        0];
    MethodsExamined=MethodsName(MethodsFlag==1);
    Num_MethodsExamined=sum(MethodsFlag);
    disp("Methods to be examined are:")
    for n=1:Num_MethodsExamined
        disp(MethodsExamined(n))
    end
    
    evaMethod='RMSE';
    
    cVec = 'bgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmy';
    pVec='.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p';

        
    meanRMSEs=zeros(Num_MethodsExamined,point_number_group);
    varRMSEs=zeros(Num_MethodsExamined,point_number_group);
    meanRMSE2s=zeros(Num_MethodsExamined,point_number_group);
    varRMSE2s=zeros(Num_MethodsExamined,point_number_group);
%%
rngNum=30;
everyAgentsSampleNum=sampleNums(end)*1.3;
predictionEvaluationDataLoad2
maxIter=20;
M=maxM;
for i=point_number_group:-1:1
    datasetNum=sampleNums(i);
    % Prepare
    for abbv=1
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
        [D,N_newX]=size(newX);
        plotFlag=1;
        fig_export_pix=300;
        eps_export=0;
        png_export=1;
        contourFlag=0;

        overlapFlag=1; %0. k-means, 1. fcm
        theta=[5,1,1]';
        Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
        savePlot=0;
    end
    
    %% delete points number
    X=[];
    Z=[];
    for m=1:M
        Agents(m).X=Agents(m).X(:,1:datasetNum);
        X=[X, Agents(m).X];
        Agents(m).Z=Agents(m).Z(1:datasetNum,:);
        Z=[Z;Agents(m).Z];
        Agents(m).N_m=datasetNum;
        subSize(m)=datasetNum;
        Agents(m).distX1=dist(Agents(m).X(1,:)).^2;
        Agents(m).distX2=dist(Agents(m).X(2,:)).^2;
        Agents(m).distXd=zeros(subSize(m),subSize(m),inputDim);
        for d=1:inputDim
            Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2;
        end
        
    end

    %% other prepare
    meanRMSE=zeros(Num_MethodsExamined,1);
    varRMSE=zeros(Num_MethodsExamined,1);
    meanRMSE2=zeros(Num_MethodsExamined,1);
    varRMSE2=zeros(Num_MethodsExamined,1);
    times=zeros(Num_MethodsExamined,1);
    
    %% Evaluate Full GP
    method='Full';
    disp('Now calculating Full GP')
    
    tic
    [Mean_total,Uncertainty_total] = GPR_predict(X,Z',theta,[range_x1;range_x2],sigma_n,plotFlag,[reso_x,reso_y]);
    toc
    realMean=Mean_total;
    realMean=reshape(realMean,1,reso_x*reso_y);
    realVar=Uncertainty_total;
    realVar=reshape(realVar,1,reso_x*reso_y);
    
    %% Evaluate Other methods
        subMeans_precal=zeros(M,N_newX);
        subVars_precal=zeros(M,N_newX);

        parfor m=1:M
            [subMeans_precal(m,:),subVars_precal(m,:)]=subGP(Agents(m),newX,sigma_n);
        end

        for n=1:Num_MethodsExamined
            clear mean_1 var_1 mean_2 var_2 Means Vars Means2 Vars2
            outputPDMM_DTCF_compare=0;
            method=MethodsExamined(n);
            tic
            switch method
                case DECNAME
                    %                     disp(method)
                    A=A_full(1:M,1:M);
                    [Means,Vars,mean_1,var_1] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n,subMeans_precal,subVars_precal,'PDMM');

                    mean_2=[];
                    var_2=[];
                    if method~="DEC-NPAE"
                        toc
                        [Means2,Vars2,mean_2,var_2] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n,subMeans_precal,subVars_precal,'DTCF');
                        outputPDMM_DTCF_compare=1;
                    end
                case NNNAME
                    %                     disp(method)
                    [mean_1,var_1] = GPR_predict_NN(Agents,method,newX,sigma_n,subMeans_precal,subVars_precal);
                    mean_2=[];
                    var_2=[];
                    meanNN=mean_1;
                    varNN=var_1;
                case NOAGNAME
%                     disp(method)
                    [mean_1,var_1] = GPR_predict_NoAg(Agents,newX,sigma_n);
                    mean_2=[];
                    var_2=[];
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
                        [mean_temp,var_temp] = GPR_predict_NN(Agents,method1,newX,sigma_n);
                        [~,~,mean_1,var_1] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,mean_temp,var_temp,'PDMM');
                        [~,~,mean_2,var_2] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,mean_temp,var_temp,'DTCF');
                    elseif NNFlag==1
                        [~,~,mean_1,var_1] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,meanNN,varNN,'PDMM');
                        [~,~,mean_2,var_2] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,meanNN,varNN,'DTCF');
                    end
                    
                case CENTER
                    disp(method)
                    disp("!!!!!!!!!!!!!!!!!Evaluation Under Construction!!!!!!!!!!!!!!!!!!!!")
            end
            toc
            TOC=toc;
            times(n,1)=TOC;
            [meanRMSE(n,1),varRMSE(n,1)] = ...
                evaluatePredictionPerformanceMetrices(realMean,realVar,mean_1,var_1,evaMethod);
            if ~isempty(mean_2)
                [meanRMSE2(n,1),varRMSE2(n,1)] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,mean_2,var_2,evaMethod);
                clear mean_2 var_2
            else
                [meanRMSE2(n,1),varRMSE2(n,1)] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,mean_1,var_1,evaMethod);
            end
            if outputPDMM_DTCF_compare==1
                gcf=figure;
                [pm2,pv2] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,Means2,Vars2,'consensusRMSE');
                [pm,pv] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,Means,Vars,'consensusRMSE');
                tiledlayout(2,1,'Padding','none','TileSpacing','compact');
                nexttile(1);
                plot(pm2),hold on,plot(pm),title('Mean Error'),hold off;
                legend(['DTCF';'PDMM'],'Location','northoutside','Orientation','horizontal')
                set(gca,'XScale','log','YScale','log');
                xlabel('iterations')
                ylabel('consensus error')
                nexttile(2);
                plot(pv2),hold on,plot(pv),title('Variance Error'),hold off;
                set(gca,'XScale','log','YScale','log');
                xlabel('iterations')
                ylabel('consensus error')
                
                sgtitle(strcat(method,' Mean and Variance consensus error'));
                fname=strcat('results\Agg\PerformanceEva2\',method,'_expRep_',num2str(exp_r_id),'_a_',num2str(M),'_maxIter_',num2str(maxIter),'_PDMM_DTCF_Compare');
                s=hgexport('factorystyle');
                s.Format='eps';
                s.FontSizeMin=10;
                s.Resolution=600;
                s.width=8;
                s.height=6;
                hgexport(gcf,fname,s);
                s.Format='png';
                hgexport(gcf,fname,s);
                close 
            end
        end
meanRMSEs(i,:)=meanRMSE;
varRMSEs(i,:)=varRMSE;
meanRMSE2s(i,:)=meanRMSE2;
varRMSE2s(i,:)=meanRMSE2;
end


return;
    %%
for exp_r_id=1:point_number_group
    rngNum=exp_r_id*10;
    everyAgentsSampleNum=sampleNums(exp_r_id);
    predictionEvaluationDataLoad
    meanRMSE=zeros(Num_MethodsExamined,1);
    varRMSE=zeros(Num_MethodsExamined,1);
    meanRMSE2=zeros(Num_MethodsExamined,1);
    varRMSE2=zeros(Num_MethodsExamined,1);
    %% Pre-exp pars
    for abbv=1
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
        [D,N_newX]=size(newX);
        plotFlag=1;
        fig_export_pix=300;
        eps_export=0;
        png_export=1;
        contourFlag=0;

        overlapFlag=1; %0. k-means, 1. fcm
        theta=[5,1,1]';
        Topology_method=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
        savePlot=0;
    end

    maxIter=20;
    
    meanRMSE=zeros(Num_MethodsExamined,1);
    varRMSE=zeros(Num_MethodsExamined,1);
    meanRMSE2=zeros(Num_MethodsExamined,1);
    varRMSE2=zeros(Num_MethodsExamined,1);
    graphs=cell(1,1);
    times=zeros(Num_MethodsExamined,1);
    
    %% Evaluate Full GP
    method='Full';
    disp('Now calculating Full GP')
    
    tic
    [Mean_total,Uncertainty_total] = GPR_predict(X,Z,theta,[range_x1;range_x2],sigma_n,plotFlag,[reso_x,reso_y]);
    toc
    realMean=Mean_total;
    realMean=reshape(realMean,1,reso_x*reso_y);
    realVar=Uncertainty_total;
    realVar=reshape(realVar,1,reso_x*reso_y);
    
    %% Evaluate Others
    
    rng(990611,'twister')
        commuRange=[maxRange,maxRange/5,maxRange/5,maxRange/5,maxRange/8];
        commuRange(tempFlag==0)=[];
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
            A=A_full;
            G=graph(A_full);
            Lap = laplacian(G);
            [~,V,~]=eig(full(Lap));
            V=diag(V);
            V=sort(V,'ascend');
            if V(2)>1e-5
                connected=1;
            else
                commuRange(expId)=commuRange(expId)*1.1;
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
        
        subMeans_precal=zeros(M,N_newX);
        subVars_precal=zeros(M,N_newX);

        parfor m=1:M
            [subMeans_precal(m,:),subVars_precal(m,:)]=subGP(Agents(m),newX,sigma_n);
        end

        for n=1:Num_MethodsExamined
            clear mean_1 var_1 mean_2 var_2 Means Vars Means2 Vars2
            outputPDMM_DTCF_compare=0;
            method=MethodsExamined(n);
            tic
            switch method
                case DECNAME
                    %                     disp(method)
                    A=A_full(1:M,1:M);
                    [Means,Vars,mean_1,var_1] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n,subMeans_precal,subVars_precal,'PDMM');

                    mean_2=[];
                    var_2=[];
                    if method~="DEC-NPAE"
                        toc
                        [Means2,Vars2,mean_2,var_2] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n,subMeans_precal,subVars_precal,'DTCF');
                        outputPDMM_DTCF_compare=1;
                    end
                case NNNAME
                    %                     disp(method)
                    [mean_1,var_1] = GPR_predict_NN(Agents,method,newX,sigma_n,subMeans_precal,subVars_precal);
                    mean_2=[];
                    var_2=[];
                    meanNN=mean_1;
                    varNN=var_1;
                case NOAGNAME
%                     disp(method)
                    [mean_1,var_1] = GPR_predict_NoAg(Agents,newX,sigma_n);
                    mean_2=[];
                    var_2=[];
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
                        [mean_temp,var_temp] = GPR_predict_NN(Agents,method1,newX,sigma_n);
                        [~,~,mean_1,var_1] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,mean_temp,var_temp,'PDMM');
                        [~,~,mean_2,var_2] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,mean_temp,var_temp,'DTCF');
                    elseif NNFlag==1
                        [~,~,mean_1,var_1] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,meanNN,varNN,'PDMM');
                        [~,~,mean_2,var_2] = GPR_predict_dec(Agents,method2,newX,A,maxIter,sigma_n,meanNN,varNN,'DTCF');
                    end
                    
                case CENTER
                    disp(method)
                    disp("!!!!!!!!!!!!!!!!!Evaluation Under Construction!!!!!!!!!!!!!!!!!!!!")
            end
            toc
            TOC=toc;
            times(n,expId)=TOC;
            [meanRMSE(n,expId),varRMSE(n,expId)] = ...
                evaluatePredictionPerformanceMetrices(realMean,realVar,mean_1,var_1,evaMethod);
            if ~isempty(mean_2)
                [meanRMSE2(n,expId),varRMSE2(n,expId)] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,mean_2,var_2,evaMethod);
                clear mean_2 var_2
            else
                [meanRMSE2(n,expId),varRMSE2(n,expId)] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,mean_1,var_1,evaMethod);
            end
            if outputPDMM_DTCF_compare==1
                gcf=figure;
                [pm2,pv2] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,Means2,Vars2,'consensusRMSE');
                [pm,pv] = ...
                    evaluatePredictionPerformanceMetrices(realMean,realVar,Means,Vars,'consensusRMSE');
                tiledlayout(2,1,'Padding','none','TileSpacing','compact');
                nexttile(1);
                plot(pm2),hold on,plot(pm),title('Mean Error'),hold off;
                legend(['DTCF';'PDMM'],'Location','northoutside','Orientation','horizontal')
                set(gca,'XScale','log','YScale','log');
                xlabel('iterations')
                ylabel('consensus error')
                nexttile(2);
                plot(pv2),hold on,plot(pv),title('Variance Error'),hold off;
                set(gca,'XScale','log','YScale','log');
                xlabel('iterations')
                ylabel('consensus error')
                
                sgtitle(strcat(method,' Mean and Variance consensus error'));
                fname=strcat('results\Agg\PerformanceEva\',method,'_expRep_',num2str(exp_r_id),'_a_',num2str(M),'_maxIter_',num2str(maxIter),'_PDMM_DTCF_Compare');
                s=hgexport('factorystyle');
                s.Format='eps';
                s.FontSizeMin=10;
                s.Resolution=600;
                s.width=8;
                s.height=6;
                hgexport(gcf,fname,s);
                s.Format='png';
                hgexport(gcf,fname,s);
                close 
            end
        end
        
    end
    
    
gcf=figure;
for m=1:Num_expGroup
    subplot(1,Num_expGroup,m);
    plot(graphs{m});
end
fname=strcat('results\Agg\PerformanceEva\',method,'_expRep_',num2str(exp_r_id),'_a_',num2str(M),'_maxIter_',num2str(maxIter),'_Graph');
saveas(gcf,fname,'png');
close gcf;

save_txt=strcat('evalueResult_',num2str(exp_r_id),'.mat');
save(save_txt);
meanRMSEs(:,:,exp_r_id)=meanRMSE;
varRMSEs(:,:,exp_r_id)=varRMSE;
meanRMSE2s(:,:,exp_r_id)=meanRMSE2;
varRMSE2s(:,:,exp_r_id)=meanRMSE2;
end
meanRMSE=mean(meanRMSEs,3);
varRMSE=mean(meanRMSEs,3);
meanRMSE2=mean(meanRMSE2s,3);
varRMSE2=mean(meanRMSE2s,3);
%% plot result
s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=800;
% Plot Graph Structures
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
for m=1:Num_expGroup
    subplot(1,Num_expGroup,m);
    plot(graphs{m});
end
fname='results/Agg/PerformanceEva/Graphs';
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;
% Plot PDMM based methods mean RMSE
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,meanRMSE(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE of predictive means - PDMM')
fname='results/Agg/PerformanceEva/MeanRMSE_PDMM';
xlabel('number of agents');
ylabel('RMSE')
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;
disp('meanRMSE1 PDMM saved')

% Plot DTCF based methods
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
clear legendTxt
legendTxt=cell(Num_MethodsExamined,1);
% meanRMSE2(isnan(meanRMSE2))=0;
% meanRMSE2((end-2):end,:)=meanRMSE((end-2):end,:);

for m=1:Num_MethodsExamined
    plot(Ms,meanRMSE2(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
% legendTxt=legendTxt{1:size(meanRMSE2,1)};
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE of predictive means - DTCF')
fname='results/Agg/PerformanceEva/MeanRMSE2_DTCF';
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;
disp('meanRMSE2 DTCF saved')
%%
% Compare mean RMSE of DTCF and PDMM
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
clear legendTxt
legendTxt=cell(1,1);
lgdCount=0;
for m=1:Num_MethodsExamined
    if any(strcmp(IANAME,MethodsExamined(m)))
        plot(Ms,meanRMSE(m,:),strcat('-',pVec(m)),'Color',cVec(m));
        lgdCount=lgdCount+1;
        legendTxt{lgdCount}=strcat(MethodsExamined(m)," PDMM");
        plot(Ms,meanRMSE2(m,:),strcat('--',pVec(m)),'Color',cVec(m));
        lgdCount=lgdCount+1;
        legendTxt{lgdCount}=strcat(MethodsExamined(m)," DTCF");
    end
end

set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE comparison of DTCF and PDMM - predictive means')
fname='results/Agg/PerformanceEva/Mean_PDMM_DTCF_Cmp';
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;
disp('PDMM DTCF compare saved')

%%
% Plot PDMM based methods variance RMSE
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
clear legendTxt
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,varRMSE(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE of predictive variances - PDMM')
fname='results/Agg/PerformanceEva/VarRMSE_PDMM';
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;

gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
clear legendTxt
legendTxt=cell(size(varRMSE2,1),1);
% varRMSE2((end-2):end,:)=varRMSE((end-2):end,:);
for m=1:size(varRMSE2,1)
    plot(Ms,varRMSE2(m,:),strcat('-',pVec(m)));
    legendTxt{m}=MethodsExamined(m);
end
legendTxt=legendTxt{1:size(varRMSE2,1)};
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE of predictive variances - DTCF')
fname='results/Agg/PerformanceEva/VarRMSE2_DTCF';
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;

%%
% Compare var RMSE of DTCF and PDMM
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on;
clear legendTxt
legendTxt=cell(1,1);
lgdCount=0;
for m=1:Num_MethodsExamined
    if any(strcmp(IANAME,MethodsExamined(m)))
        plot(Ms,varRMSE(m,:),strcat('-',pVec(m)),'Color',cVec(m));
        lgdCount=lgdCount+1;
        legendTxt{lgdCount}=strcat(MethodsExamined(m)," PDMM");
        plot(Ms,varRMSE2(m,:),strcat('--',pVec(m)),'Color',cVec(m));
        lgdCount=lgdCount+1;
        legendTxt{lgdCount}=strcat(MethodsExamined(m)," DTCF");
    end
end

set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
sgtitle('RMSE comparison of DTCF and PDMM - predictive variances')
fname='results/Agg/PerformanceEva/Var_PDMM_DTCF_Cmp';
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;
disp('PDMM DTCF variance compare saved')


%%
gcf=figure;
tiledlayout(1,1,'TileSpacing','compact','Padding','none');
nexttile(1);
hold on
legendTxt=cell(Num_MethodsExamined,1);
for m=1:Num_MethodsExamined
    plot(Ms,times(m,:),'-o')
    legendTxt{m}=MethodsExamined(m);
end
set(gca, 'YScale', 'log');
legend(legendTxt,'Location','northoutside','Orientation','horizontal','NumColumns',3);
hold off
title('Time used')
fname='results/Agg/PerformanceEva/Times';
saveas(gcf,fname,'png');
s.Format='png';
hgexport(gcf,fname,s);
s.Format='eps';
hgexport(gcf,fname,s);
close gcf;

end_time=datetime('now');
disp(strcat("The code ends at ",datestr(end_time)))
code_duration=end_time-start_time;
disp(strcat("The code duration is ",string(code_duration)))
%%
function [Mean,Var]=subGP(Agent,newX,sigma_n)
if nargin==2
    sigma_n=0;
end

N_newX=size(newX,2);

X=Agent.X;
Z=Agent.Z;
[m,n]=size(Z);
if m<n
    Z=Z';
end
theta=[Agent.sigma_f;Agent.l];

K=getK(X,theta,sigma_n);

invK=inv(K);
L=chol(K)';
alpha=L'\(L\Z);

Mean=zeros(N_newX,1);
Var=zeros(N_newX,1);

parfor n=1:N_newX
    x_star=newX(:,n);
    [Mean(n),Var(n)] = GPRSinglePointPredict(X,x_star,alpha,L,theta,sigma_n);
end

end