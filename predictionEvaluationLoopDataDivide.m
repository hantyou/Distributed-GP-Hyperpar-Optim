function [Agents,clusterIdx] = predictionEvaluationLoopDataDivide(range,X,Z,M,overlapFlag)


            cVec = 'bgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmybgrcmy';
            pVec='.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p.*o+xsd^p';
X=X';
[sampleSize,D]=size(X);

numC=M;
if overlapFlag==1
    fcm_option=[3,50,1e-6,false];
    [centers,U] = fcm(X,numC,fcm_option);
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
    [idx,C] = kmeans(X,numC,'Distance','cityblock',...
        'Replicates',5,'Options',opts);
end
Agents_Posi=C';
clusterIdx=idx;


figure;
hold on
LegendTxt=cell(numC+1,1);
for id=1:numC
    plot(X(idx==id,1),X(idx==id,2),[cVec(id) pVec(id)],'MarkerSize',10)
    LegendTxt{id}=strcat('Cluster',num2str(id));
end
title 'Cluster Assignments and Centroids'
LegendTxt{end}='Centroids';
plot(C(:,1),C(:,2),'kx',...
    'MarkerSize',15,'LineWidth',3)
legend(LegendTxt,'Location','NW')
hold off


subSize=ones(M,1);
X=X';
subSize=zeros(M,1);
for m=1:M
    subSize(m)=length(idx(idx==m));
    sampleIdx=cumsum(subSize)';
    sampleIdx=[0,sampleIdx];
end


sampleSize=sum(subSize);

%% Dataset division and agents initialization
% agents initialization
% Distribute according to range of detection
localDataSetsSize=subSize;
idx1=1:sampleSize;
idx=ones(M,max(subSize));
%     clusterIdx=clusterIdx';
for m=1:M
        idx(m,1:subSize(m))=find(clusterIdx==m)';
end
idxedZ=Z(idx);
subDataSetsZ=idxedZ;
inputDim=size(X,1);
idxedX=zeros(inputDim,max(subSize),M);
for m=1:M
    idxedX(:,:,m)=X(:,idx(m,:));
end
subDataSetsX=idxedX;
% Generate agents
subSize;
Agents=agent.empty(M,0);
for m=1:M
    Agents(m).Code=m;
    Agents(m).Z=subDataSetsZ(m,1:subSize(m))';
    Agents(m).X=subDataSetsX(:,1:subSize(m),m);
    Agents(m).idx=idx(m,1:subSize(m));
    Agents(m).N_m=localDataSetsSize(m);
    Agents(m).M=M;
    Agents(m).action_status=1;
    %     Agents(m).commuRange=2.5;

    Agents(m).distX1=dist(Agents(m).X(1,:)).^2;
    Agents(m).distX2=dist(Agents(m).X(2,:)).^2;
    Agents(m).distXd=zeros(subSize(m),subSize(m),inputDim);
    Agents(m).Position=Agents_Posi(:,m);
    for d=1:inputDim
        Agents(m).distXd(:,:,d)=dist(Agents(m).X(d,:)).^2;
    end
end

end

