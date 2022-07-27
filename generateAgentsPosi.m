function [Agents_Posi,X,subSize]= generateAgentsPosi(Method,range,M,Agents_measure_range,everyAgentsSampleNum,overlap)
%GENERATEAGENTSPOSI Summary of this function goes here
%   Method: 0. totally random; 1. equal generation
%   X, subSize: When Method = 0, [] output for these two variables
%   If there is reasonable X output, the X is a cell of size M x 1
%% Input Interface
switch nargin
    case 2
        M=8;
        disp('Default agents num set to 8')
        Agents_measure_range=min(range(1,2)-range(1,1),range(2,2)-range(2,1));
        Agents_measure_range=Agents_measure_range/3;
        txtShow=strcat("Default measurement range set to range_min/3 = ",...
            num2str(Agents_measure_range));
        disp(txtShow)
        clear txtShow
        everyAgentsSampleNum=70;
        disp('Default points num/agent set to 70')
        overlap=1;
        disp('Geomatric overlap allowed')
        
    case 3
        Agents_measure_range=min(range(1,2)-range(1,1),range(2,2)-range(2,1));
        Agents_measure_range=Agents_measure_range/3;
        txtShow=strcat("Default measurement range set to range_min/3 = ",...
            num2str(Agents_measure_range));
        disp(txtShow)
        clear txtShow
        everyAgentsSampleNum=70;
        disp('Default points num/agent set to 70')
        overlap=1;
        disp('Geomatric overlap allowed')
    case 4
        everyAgentsSampleNum=70;
        disp('Default points num/agent set to 70')
        overlap=1;
        disp('Geomatric overlap allowed')
    case 5
        overlap=1;
        disp('Geomatric overlap allowed')
end
%% Generate agents positions
range_x1=range(1,:);
range_x2=range(2,:);
scaling_factor=0.9; % prevent agents from being generated on the edge

if Method==0
    X=[];
    subSize=[];
    Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,M)*scaling_factor;
        unifrnd(range_x2(1),range_x2(2),1,M)*scaling_factor];
else
    %             disp('also deciding sampling points');
    Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,M)*0.9;
        unifrnd(range_x2(1),range_x2(2),1,M)*0.9];
    subSize=ones(M,1)*everyAgentsSampleNum;
    [X_temp,subSize,~] = decideSamplePoints(1,subSize,range,Agents_Posi,Agents_measure_range);

    X_temp=X_temp';
    cVec = 'bgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmy';
    pVec='.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p';

    numC=M;
    % Devide datasets
    % Based on overlap or non-overlap option
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
%     clusterIdx=idx;
    % Put X in a cell
    X=cell(M,1);
    subSize=zeros(M,1);
    for m=1:M
        X{m}=X_temp(idx==m,:)';
        [~,subSize(m)]=size(X{m});
    end

    % Plot
    gcf=figure;
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
    close gcf;
end


%% Output Interface
switch nargout
    case 1
        X=[];
        subSize=[];
    case 2
        subSize=[];
end
end

