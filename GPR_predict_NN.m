function [Mean,Var] = GPR_predict_NN(Agents,method,newX,sigma_n)
%GPR_PREDICT_DEC Summary of this function goes here
%   Detailed explanation goes here
if nargin==3
    sigma_n=0;
end

M=length(Agents);
N_newX=size(newX,2);

subMeans=zeros(M,N_newX);
subVars=zeros(M,N_newX);

parfor m=1:M
    [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
end


switch upper(method)
    case {'NN-POE'}
    case {'NN-GPOE'}
    case {'NN-BCM'}
    case {'NN-GRBCM'}
    case {'NN-NPAE'}
        disp('NN-NPAE')
        [Mean,Var]=NN_NPAE(Agents,newX,subMeans,subVars,sigma_n);
        disp('NN-NPAE')
end

end


function [mu,var]=NN_NPAE(Agents,newX,subMean,subVar,sigma_n)
M=length(Agents);
[Xs,Covs,invCovs]=getNeighborCovMatrix(Agents);

[D,N_newX]=size(newX);
mu=zeros(M,N_newX);
var=zeros(M,N_newX);

for m=1:M
    all_ind=[m,Agents(m).Neighbors];
    subMean_m=subMean(all_ind,:);
    subVar_m=subVar(all_ind,:);
    Covs_m=Covs{m};
    N_size=Agents(m).N_size;
    theta=[Agents(m).sigma_f;Agents(m).l];
    k_star_star=getK(newX(:,1),theta,sigma_n);
    parfor n=1:N_newX
        newX_n=newX(:,n);
        kM_m_n=zeros(N_size+1,1);
        KM_m_n=zeros(N_size+1,N_size+1);
        for i=1:N_size+1
            kxi=getK(newX_n,Xs{m}{i},theta,sigma_n);
            kM_m_n(i)=kxi*invCovs{m}{i}*kxi';
            for j=i:N_size+1
                kxj=getK(newX_n,Xs{m}{j},theta,sigma_n);
                cov_ij=Covs{m}{i,j};
                KM_m_n(i,j)=kxi*invCovs{m}{i}*cov_ij*invCovs{m}{j}*kxj';
                KM_m_n(j,i)=KM_m_n(i,j)';
            end
        end
        mu(m,n)=kM_m_n'/KM_m_n*subMean_m(:,n);
        var(m,n)=k_star_star-kM_m_n'/KM_m_n*kM_m_n;
    end
end

end

function [Xs,Covs,invCovs]=getNeighborCovMatrix(Agents)
M=length(Agents);
Xs=cell(1,M);
Covs=cell(1,M);
invCovs=cell(1,M);
for m=1:M
    N_size=Agents(m).N_size;
    Xs{m}=cell(N_size+1,1);
    Covs{m}=cell(N_size+1,N_size+1);
    invCovs{m}=cell(N_size+1,1);

    Xs{m}{1}=Agents(m).X;
    for n=1:N_size
        Xs{m}{n+1}=Agents(Agents(m).Neighbors(n)).X;
    end

    theta=[Agents(m).sigma_f;Agents(m).l];
    for i=1:N_size+1
        for j=1:N_size+1
            Covs{m}{i,j}=getK(Xs{m}{i},Xs{m}{j},theta,Agents(m).sigma_n);
            if i==j
                invCovs{m}{i}=inv(Covs{m}{i,j});
            end
        end
    end
end

end

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
theta=Agent.z;

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
