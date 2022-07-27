clc,clear;
close all;
start_time=datetime('now');
disp(strcat("The code begins at ",datestr(start_time)))

if usejava('desktop')
    figureVisibility='off';
else
    figureVisibility='off';
end
set(0,'DefaultFigureVisible',figureVisibility)
% %%
% delete(gcp('nocreate'))
% try
%     parpool(30);
% catch
%     parpool(8)
% end
%% Evaluation setup
range_x1=[-5,5];
range_x2=[-5,5];
range=[range_x1;range_x2];
M=8;
AgentCommuDist=4;
reso_x=256;
reso_y=256;
everyAgentsSampleNum=70;
Agents_measure_range=3;
samplingMethod=2; % 1. uniformly distirbuted accross region; 2. near agents position, could lose some points if out of range
agentsScatterMethod=2; % 1. Randomly distributed accross the area; 2. K_means center
reso=[reso_x,reso_y];
repeatNum=3;
theta=[5;1;1];
sigma_n=sqrt(0.1);
maxRange=max(range(:))-min(range(:));
rngNum=11;
D=2;

cVec = 'bgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmy';
% pVec='.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p';
pVec='*o+xsd^p*o+xsd^p*o+xsd^p*o+xsd^p*o+xsd^p*o+xsd^p';

%% data loading
rng(rngNum,"twister");
realDataSet=0;
overlap=1; % 1. overlap allowed (fuzzy c-means), 2. disjoint clusters
[F_true,range,reso] = standard_field_loading(range,reso,realDataSet,theta);

[reso_x,reso_y]=size(F_true);
range_x1=range(1,:); range_x2=range(2,:);
% generate grid for later use
t_x=linspace(range_x1(1),range_x1(2),reso_x);
t_y=linspace(range_x2(1),range_x2(2),reso_y);
[mesh_x1,mesh_x2]=meshgrid(t_x,t_y);

%% Decide agents position
AgentPosiMethod=1; % 0. totally random; 1. equal generation
%  [Agents_Posi,X,subSize]= generateAgentsPosi(Method,range,M,everyAgentsSampleNum,overlap)
%   Method: 0. totally random; 1. equal generation
%   X, subSize: When Method = 0, [] output for these two variables
[Agents_Posi,Xs,subSize]= generateAgentsPosi(AgentPosiMethod,range,M,...
    Agents_measure_range,everyAgentsSampleNum,overlap);
%% Decide sampling points position
% if not yet decided
SamplingPointsMethod=2;
independentSamplingPoints=1;
if independentSamplingPoints
    subSize=ones(M,1)*everyAgentsSampleNum;
    [X_temp,subSize,sampleIdx] = decideSamplePoints(SamplingPointsMethod,...
        subSize,range,Agents_Posi,Agents_measure_range);
    Xs=cell(M,1);
    for m=1:M
        Xs{m}=X_temp(:,sampleIdx(m)+1:sampleIdx(m+1));
    end
    clear X_temp sampleIdx
end
% Collect Xs into global dataset
X=[];
for m=1:M
    X=[X,Xs{m}];
end
%% Take measurement values
Ys=cell(M,1); % Clean samples, cell structure
Y=[];
Zs=cell(M,1); % Noisy samples, cell structure
Z=[];
for m=1:M
    Ys{m}=interp2(mesh_x1,mesh_x2,F_true,Xs{m}(1,:),Xs{m}(2,:));
    Y=[Y,Ys{m}];
    Zs{m}=Ys{m}+sigma_n*randn(size(Ys{m}));
    Z=[Z,Zs{m}];
end
%% Agents target initialization
Agents=agent.empty(M,0);
for m=1:M
    Agents(m).Code=m;
    Agents(m).Z=Zs{m}';
    Agents(m).X=Xs{m};
    Agents(m).N_m=subSize(m);
    Agents(m).M=M;
    Agents(m).commuRange=AgentCommuDist;

    Agents(m).distX1=dist(Agents(m).X(1,:)).^2;
    Agents(m).distX2=dist(Agents(m).X(2,:)).^2;
    Agents(m).distXd=zeros(subSize(m),subSize(m),D);
    Agents(m).Position=Agents_Posi(:,m);
    for d=1:D
        Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2;
    end
end
%% Generate Topology
TopologyMethod=2; % 1: stacking squares; 2: nearest link with minimum link; 3: No link
[A_full,Agents]=generateTopology(Agents,TopologyMethod);
G=graph(A_full);
gcf=figure;
plot(G,'b','XData',Agents_Posi(1,:),'YData',Agents_Posi(2,:));
close gcf;
%% Field Setting In One Figure
for i=1
    gcf=figure;
    tiledlayout(1,2,"Padding","none",'TileSpacing','compact');
    gca1=nexttile(1); % draw field
    surf(t_x,t_y,F_true,'EdgeColor','none','FaceAlpha',0.8);
    maxValue_mean=max(F_true(:));
    minValue_mean=min(F_true(:));
    colormap('gray');
    set(gca,"CLim",[minValue_mean,maxValue_mean]);
    % colorbar;
    pbaspect([1 1 1])
    set(gca,"YDir","normal");
    [colors,pc]=pcsel(Agents_Posi',5);
    title("Field shown in 3D")
    gca2=nexttile(2); % draw field and samples
    hold on;
    imagesc(t_x,t_y,F_true);
    maxValue_mean=max(F_true(:));
    minValue_mean=min(F_true(:));
    colormap(gca,'gray');
    set(gca,"CLim",[minValue_mean,maxValue_mean]);
    colorbar;
    set(gca,"YDir","normal");
    sc=zeros(M,1);
    ap=zeros(M,1);
    for m=1:M
        sc(m)=scatter(Agents(m).X(1,:),Agents(m).X(2,:),...
            25,strcat(pc{colors(m)}(1),pVec(m)));
    end
    % for m=1:M
    %     scatter(Agents(m).Position(1),Agents(m).Position(2),100,strcat(cVec(m),'^'),"filled");
    % end
    pg=plot(G,'b','XData',Agents_Posi(1,:),...
        'YData',Agents_Posi(2,:),'LineWidth',2);
    for m=1:M
        highlight(pg, m, 'NodeColor',pc{colors(m)}(1));
    end
    ylim(range_x2)
    xlim(range_x1)
    pbaspect([1 1 1])
    title("Field and sampling inputs");
    hold off
    s=hgexport('factorystyle');
    s.Resolution=600;
    s.Width=8;
    s.Height=4;
    s.Format='png';
    mkdir('results','InducingCompare');
    fname='results/InducingCompare/ExpSetting';
    hgexport(gcf,fname,s);
    close gcf
    %
    gcf=figure;
    tiledlayout(1,1,"Padding","none",'TileSpacing','compact');
    nexttile(1);
    surf(t_x,t_y,F_true,'EdgeColor','none','FaceAlpha',0.8);
    maxValue_mean=max(F_true(:));
    minValue_mean=min(F_true(:));
    colormap('gray');
    set(gca,"CLim",[minValue_mean,maxValue_mean]);
    colorbar;
    pbaspect([1 1 1])
    set(gca,"YDir","normal");
%     [colors,pc]=pcsel(Agents_Posi',5);
    title("Field shown in 3D");
    s=hgexport('factorystyle');
    s.Resolution=600;
    scaleFig=0.8;
    s.Width=4.5*scaleFig;
    s.Height=4*scaleFig;
    s.Format='png';
    s.FontSizeMin=11;
    fname='results/InducingCompare/Field3D';
    hgexport(gcf,fname,s);
    close gcf
    %
    gcf=figure;
    tiledlayout(1,1,"Padding","none",'TileSpacing','compact');
    nexttile(1);
    hold on;
    imagesc(t_x,t_y,F_true);
    maxValue_mean=max(F_true(:));
    minValue_mean=min(F_true(:));
    colormap(gca,'gray');
    set(gca,"CLim",[minValue_mean,maxValue_mean]);
    colorbar;
    set(gca,"YDir","normal");
    sc=zeros(M,1);
    ap=zeros(M,1);
    lgdTex=cell(M+2,1);
    for m=1:M
        sc(m)=scatter(Agents(m).X(1,:),Agents(m).X(2,:),...
            25,strcat(pc{colors(m)}(1),pVec(m)));
        lgdTex{m}=strcat('$\mathcal{D}_',num2str(m),'$');
    end
    pg=plot(G,'b','XData',Agents_Posi(1,:),...
        'YData',Agents_Posi(2,:),'LineWidth',2);
    lgd_node=plot(NaN,NaN,'k.','MarkerSize',18);
    lgdTex{M+1}='agents';
    lgd_edge=plot([NaN NaN],[NaN NaN],'b-','LineWidth',2);
    lgdTex{M+2}='links';
    for m=1:M
        highlight(pg, m, 'NodeColor',pc{colors(m)}(1));
    end
    ylim(range_x2)
    xlim(range_x1)
    legend([sc;lgd_node;lgd_edge],lgdTex,'Location','eastoutside','Interpreter','latex');
    pbaspect([1 1 1])
    title("Field and sampling inputs");
    hold off
    s=hgexport('factorystyle');
    s.Resolution=600;
    s.FontSizeMin=11;
    s.Width=6*scaleFig;
    s.Height=4*scaleFig;
    s.Format='png';
    fname='results/InducingCompare/FieldAndTopo';
    hgexport(gcf,fname,s);
    close gcf
end
%% Prepare for predict
pre_reso_x=100;
pre_reso_y=100;
ts_1=linspace(range_x1(1),range_x1(2),pre_reso_x);
ts_2=linspace(range_x2(1),range_x2(2),pre_reso_y);
[mesh_x,mesh_y]=meshgrid(ts_1,ts_2);
vecX=mesh_x(:);
vecY=mesh_y(:);
newX=[vecX,vecY]';
clear vecX vecY

%% Apply Inducing Point Algorithm
Scales=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
exp_num=length(Scales);
Means_ind=cell(M,exp_num);
Vars_ind=cell(M,exp_num);
Fv_values=zeros(M,exp_num);
maxValue_mean=-Inf;
minValue_mean=Inf;
for exp_id=exp_num:-1:1
    indScale=Scales(exp_id);
    tic
    disp(strcat("Now the inducing scale is ",num2str(indScale)));
    for m=1:M
        m_ind=ceil(indScale*Agents(m).N_m);
        if exp_id==exp_num
        [Agents(m).X_induced,Agents(m).Z_induced,Fvs]=...
            FindInducingPoints(Agents(m).X,Agents(m).Z',m_ind,theta,sigma_n);
        Fv_values(m,exp_id)=Fvs(m_ind);
        else
            Agents(m).X_induced=Agents(m).X_induced(:,1:m_ind);
            Agents(m).Z_induced=Agents(m).Z_induced(:,1:m_ind);
            Fv_values(m,exp_id)=Fvs(m_ind);
        end
    end
    toc
    % plot part
    for i=1
        %
        gcf=figure;
        tiledlayout(1,1,"Padding","none",'TileSpacing','compact');
        nexttile(1);
        hold on;
        imagesc(t_x,t_y,F_true);
        maxValue_mean=max(F_true(:));
        minValue_mean=min(F_true(:));
        colormap(gca,'gray');
        set(gca,"CLim",[minValue_mean,maxValue_mean]);
        colorbar;
        set(gca,"YDir","normal");
        sc=zeros(M,1);
        ap=zeros(M,1);
        lgdTex=cell(M+2,1);
        for m=1:M
            sc(m)=scatter(Agents(m).X_induced(1,:),Agents(m).X_induced(2,:),...
                25,strcat(pc{colors(m)}(1),pVec(m)));
            lgdTex{m}=strcat('$\mathcal{D}_{',num2str(m),'}^{-}$');
        end
        pg=plot(G,'b','XData',Agents_Posi(1,:),...
            'YData',Agents_Posi(2,:),'LineWidth',2);
        lgd_node=plot(NaN,NaN,'k.','MarkerSize',18);
        lgdTex{M+1}='agents';
        lgd_edge=plot([NaN NaN],[NaN NaN],'b-','LineWidth',2);
        lgdTex{M+2}='links';
        for m=1:M
            highlight(pg, m, 'NodeColor',pc{colors(m)}(1));
        end
        ylim(range_x2)
        xlim(range_x1)
        legend([sc;lgd_node;lgd_edge],lgdTex,'Location','eastoutside','Interpreter','latex');
        pbaspect([1 1 1])
        title("Field and sampling inputs");
        hold off
        s=hgexport('factorystyle');
        s.Resolution=300;
        s.FontSizeMin=11;
        s.Width=6*scaleFig;
        s.Height=4*scaleFig;
        s.Format='png';
        fname=strcat('results/InducingCompare/FieldAndInducedPoints_indScale_',num2str(indScale),'.png');
        hgexport(gcf,fname,s);
        close gcf
    end
    % Predict
    for m=1:M
        [Mean_temp,Var_temp]=...
            subGP2(Agents(m).X_induced,Agents(m).Z_induced,...
            newX,theta,sigma_n);
        maxValue_mean=max(maxValue_mean,max(Mean_temp(:)));
        minValue_mean=min(minValue_mean,min(Mean_temp(:)));
        Means_ind{m,exp_id}=reshape(Mean_temp,[pre_reso_x,pre_reso_x]);
        Vars_ind{m,exp_id}=reshape(Var_temp,[pre_reso_x,pre_reso_x]);
    end
end
%%
gcf=figure;
tiledlayout(M,exp_num,'Padding','compact','TileSpacing','compact');
for m=1:M
    for exp_id=1:exp_num
        nexttile((m-1)*exp_num+exp_id);
        imagesc(ts_1,ts_2,Means_ind{m,exp_id});
        set(gca,'CLim',[minValue_mean,maxValue_mean]);
        if m==1
            ttlTex={num2str(Scales(exp_id));num2str(Fv_values(m,exp_id))};
            title(ttlTex);
        else
            ttlTex=num2str(Fv_values(m,exp_id));
            title(ttlTex);
        end
        if exp_id==1
            ylabel(strcat("agent ",num2str(m)));
        end
    end
end
colorbar;
s=hgexport('factorystyle');
s.Resolution=200;
% s.FontSizeMin=10;
s.Width=exp_num*1.1;
s.Height=M;
s.Format='png';
s.Bounds='tight';
s.FixedFontSize=5;
s.ScaledFontSize=5;
% s.FontMode='fixed';
fname=strcat('results/InducingCompare/InducedMeanCompare');
hgexport(gcf,fname,s);
close gcf

%% Predict based on induced points
%%
disp('data loading/generation end')
disp('%%%%%%%%%%%%%%%%%%%%Examine Part Begin%%%%%%%%%%%%%%%%%%%%%%%')
close all
% clearvars -except A_full Agents Agents_Posi cVec pVec X Y Z range sampleSize sigma_n
