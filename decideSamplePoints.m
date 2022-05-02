function [X,subSize,sampleIdx] = decideSamplePoints(method,subSize,range,Agents_Posi,Agents_measure_range)
%DECIDESAMPLEPOINTS Summary of this function goes here
%   Detailed explanation goes here
M=length(subSize);

sampleSize=sum(subSize);
sampleIdx=cumsum(subSize)';
sampleIdx=[0,sampleIdx];
range_x1=range(1,:);
range_x2=range(2,:);

% Decide the sample points positions of each agent
if method==2
    X=zeros(2,sampleSize);
    sampleIdx=cumsum(subSize)';
    sampleIdx=[0,sampleIdx];
    for m=1:M
        rho=sqrt(rand(1,subSize(m)))*Agents_measure_range;
        theta=2*pi*rand(1,subSize(m));
        X1_m=Agents_Posi(1,m)+rho.*cos(theta);
        X2_m=Agents_Posi(2,m)+rho.*sin(theta);
        X_m=[X1_m;X2_m];
        X(:,sampleIdx(m)+1:sampleIdx(m+1))=X_m;
        
        
        idx=X_m(1,:)>range_x1(2);
        X_m(:,idx)=[];
        idx=X_m(1,:)<range_x1(1);
        X_m(:,idx)=[];
        
        idx=X_m(2,:)>range_x2(2);
        X_m(:,idx)=[];
        idx=X_m(2,:)<range_x2(1);
        X_m(:,idx)=[];
        subSize(m)=length(X_m(1,:));
    end
    sampleSize=sum(subSize);
    sampleIdx=cumsum(subSize)';
    sampleIdx=[0,sampleIdx];
    idx=X(1,:)>range_x1(2);
    X(:,idx)=[];
    idx=X(1,:)<range_x1(1);
    X(:,idx)=[];
    
    idx=X(2,:)>range_x2(2);
    X(:,idx)=[];
    idx=X(2,:)<range_x2(1);
    X(:,idx)=[];
    X1=X(1,:);
    X2=X(2,:);
end


if method==1
    % generate input points
    X1=unifrnd(range_x1(1),range_x1(2),1,sampleSize);
    X2=unifrnd(range_x2(1),range_x2(2),1,sampleSize);
    X=[X1;X2];
end

end

