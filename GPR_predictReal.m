function [predictY_mean,predictY_var] = GPR_predictReal(X,Z,theta,newXs)
%GPR_PREDICTREAL Summary of this function goes here
%   Detailed explanation goes here
sigma_f=theta(1);
l=theta(2:end);
N=size(X,2);
sigma_n=0.01;
% X=subDataSetsX(:,:,1);

% Z=subDataSetsZ(1,:);
% N=sampleSize/M;

K=zeros(N,N);
% sigma_f=sigma_pxADMM_fd_sync;
% l=l_pxADMM_fd_sync;
% l=newL;
parfor i=1:N
    for j=1:N
        K(i,j)=k(X(:,i),X(:,j),sigma_f,l,sigma_n);
    end
end
% K=K+sigma_n^2*rand(size(K,1),size(K,2));
L=chol(K)';
y=Z';
alpha=L'\(L\y);
newDayLength=max(newXs(3,:))-min(newXs(3,:))+1;
samplePointsNum=newDayLength;
days=linspace(1,newDayLength,samplePointsNum);
new_coors=newXs(1:2,:);
cityNum=size(new_coors,2)/newDayLength;
predictY_mean=zeros(cityNum,samplePointsNum);
predictY_var=zeros(cityNum,samplePointsNum);

%wb=waitbar(0,'Preparing','Name','Predict Cities');
%set(wb,'color','w');
for c=1:cityNum
    %waitbar(c/cityNum,wb,sprintf('%s %d','City: ',c))
% disp(c)
    for d=1:samplePointsNum
        x_star=[newXs(1:2,(c-1)*newDayLength+1);days(d);newXs(4,(c-1)*newDayLength+1)];
        [y_est,y_var_est] = GPRSinglePointPredict(X,x_star,alpha,L,theta,sigma_n);
        
        predictY_mean(c,d)=y_est;
        predictY_var(c,d)=y_var_est;
    end
end


%delete(wb);
end

