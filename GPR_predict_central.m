function [Mean,Var] = GPR_predict_central(Agents,method,newX,sigma_n)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin==3
    sigma_n=0;
end

M=length(Agents);
N_newX=size(newX,2);

subMeans=zeros(M,N_newX);
subVars=zeros(M,N_newX);

for m=1:M
    [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
end

switch method
    case {'gpoe','gPoE','GPOE','gPOE'}
        disp(method);
        beta=ones(M,1)./M;
        [Mean,Var]=gPoE(subMeans,subVars,beta);
    case {'poe','PoE','POE'}
        disp(method);
        [Mean,Var]=PoE(subMeans,subVars);
    case {'bcm','BCM'}
        disp(method);
        k_star_star=zeros(1,N_newX);
        theta=Agents(1).z;
        for m=1:M
            theta=theta+Agents(m).z;
        end
        theta=theta./M;
        for n=1:N_newX
            k_star_star(n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
        end
        [Mean,Var]=BCM(subMeans,subVars,1./k_star_star);
    case {'rbcm','rBCM','RBCM'}
        disp(method);
        k_star_star=zeros(1,N_newX);
        theta=Agents(1).z;
        for m=1:M
            theta=theta+Agents(m).z;
        end
        theta=theta./M;
        for n=1:N_newX
            k_star_star(n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
        end
        beta=0.5*(log(k_star_star)-log(subVars));
        [Mean,Var]=rBCM(subMeans,subVars,1./k_star_star,beta);
end



end

%% inner functions

function [Mean,Var]=PoE(subMean,subVar)
[M,N_newX]=size(subMean);
invVar=1./subVar;
varPoE=sum(invVar,1);
weight=invVar;
Mean=sum(subMean.*weight,1)./varPoE;
Var=1./varPoE;
end

function [Mean,Var]=gPoE(subMean,subVar,beta)
[M,N_newX]=size(subMean);
invVar=1./subVar;
varPoE=sum(beta.*invVar,1);
weight=invVar;
Mean=sum(beta.*subMean.*weight,1)./varPoE;
Var=1./varPoE;
end

function [Mean,Var]=BCM(subMean,subVar,k_star_star)
[M,N_newX]=size(subMean);
invVar=1./subVar;
varBCM=sum(invVar,1)+(1-M)*k_star_star;
weight=invVar;
Mean=sum(subMean.*weight,1)./varBCM;
Var=1./varBCM;
end

function [Mean,Var]=rBCM(subMean,subVar,k_star_star,beta)
[M,N_newX]=size(subMean);
invVar=1./subVar;
varBCM=sum(beta.*invVar,1)+(1-sum(beta,1)).*k_star_star;
weight=invVar;
Mean=sum(beta.*subMean.*weight,1)./varBCM;
Var=1./varBCM;
end

%%%

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

%%%

function K=getK(X1,X2,theta,sigma_n)
if nargin<4
    sigma_n=theta;
    theta=X2;

    X=X1;
    X2=X1;
end

sigma_f=theta(1);
l=theta(2:end);

[D,M]=size(X1);
[~,N]=size(X2);
K=zeros(M,N);

parfor i=1:M
    for j=1:N
        K(i,j)=kernelFunc(X1(:,i),X2(:,j),theta,sigma_n,'RBF');
    end
end
end

function k=kernelFunc(x1,x2,theta,sigma_n,type)
switch type
    case {'RBF','rbf'}
        sigma_f=theta(1);
        l=theta(2:end);
        k=sigma_f^2*exp(-0.5*(x1-x2)'*inv(diag(l.^2))*(x1-x2));
        if x1==x2
            k=k+sigma_n^2;
        end
end


end