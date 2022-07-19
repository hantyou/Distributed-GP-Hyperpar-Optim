function [Mean,Var] = GPR_predict_NN(Agents,method,newX,sigma_n,submean_old,subvar_old)
%GPR_PREDICT_DEC Summary of this function goes here
%   Detailed explanation goes here
if nargin==3
    sigma_n=0;
end

M=length(Agents);
N_newX=size(newX,2);


if nargin==6
    subMeans=submean_old;
    subVars=subvar_old;
elseif nargin<=4
    
    subMeans=zeros(M,N_newX);
    subVars=zeros(M,N_newX);
    
    parfor m=1:M
        [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
    end
end

switch upper(method)
    case {'NN-POE'}
    case {'NN-GPOE'}
    case {'NN-BCM'}
    case {'NN-GRBCM'}
    case {'NN-NPAE'}
        disp('NN-NPAE')
        pause(0.001);
        [Mean,Var]=NN_NPAE(Agents,newX,subMeans,subVars,sigma_n);
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
        warning('off','all');
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
        %         var(m,n)=k_star_star-kM_m_n'/KM_m_n*kM_m_n;
        var(m,n)=k_star_star-kM_m_n'/KM_m_n*(k_star_star-subVar_m(:,n));
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
function [mu,var]=NN_NPAE3(Agents,newX,subMean,subVar,sigma_n)
epsilon=0.1;
[M,N_newX]=size(subMean);
invKs=cell(M,1);
k_A=zeros(M,N_newX);
K_A=zeros(M,M,N_newX);
C_mn=cell(M,M);
k_ms=cell(M,1);
q_mu_old=zeros(M,N_newX);
q_sigma_old=zeros(M,N_newX);
q_mu_new=zeros(M,N_newX);
q_sigma_new=zeros(M,N_newX);
k_star_star=zeros(M,N_newX);
sigma_f=zeros(M,1);

for m=1:M
    sigma_f(m)=Agents(m).sigma_f;
    X1=Agents(m).X;
    theta=[Agents(m).sigma_f;Agents(m).l];
    %theta=[Agents(1).sigma_f;Agents(1).l];
    for n=1:N_newX
        k_star_star(m,n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
    end
    for n=1:M
        C_mn{m,n}=getK(X1,Agents(n).X,theta,sigma_n);
    end
    k_ms{m}=getK(newX,Agents(m).X,theta,sigma_n); % N_newX * agent_N_data
    invKs{m}=inv(getK(Agents(m).X,theta,sigma_n)); % Ks{m} can be accessed by all agents.
end
% K_A_m=cell(M,1);

for m=1:M
    %     K_A_m{m}=zeros(1,M,N_newX);
    K_A_m=zeros(M,N_newX);
    %     theta=[Agents(m).sigma_f;Agents(m).l];
    k_m=k_ms{m};
    %     k_A(m,:)=diag(k_m*invKs{m}*k_m');
    invKs_m=invKs{m};
    parfor n=1:N_newX
        k_A(m,n)=k_m(n,:)*invKs_m*k_m(n,:)';
    end
    for n=1:M
        %     for n=[m,Agents(m).Neighbors]
        k_n=k_ms{n};
        
        parfor num=1:N_newX
            K_A_m(n,num)=k_m(num,:)*invKs_m*C_mn{m,n}*invKs{n}*k_n(num,:)';
            K_A(m,n,num)=k_m(num,:)*invKs_m*C_mn{m,n}*invKs{n}*k_n(num,:)';
        end
        
    end
    % pre initial JOR
    q_mu_old(m,:)=subMean(m,:)./K_A_m(m,:);
    q_sigma_old(m,:)=k_A(m,:)./K_A_m(m,:);
    %     K_A(m,:,:)=K_A_m;
end
% JOR initial
q_mu_direct=zeros(M,N_newX);
q_sigma_direct=zeros(M,N_newX);
parfor n=1:N_newX
    warning('off','all');
    q_mu_direct(:,n)=pinv(K_A(:,:,n))*subMean(:,n);
    q_sigma_direct(:,n)=pinv(K_A(:,:,n))*k_A(:,n);
end
maxIterJOR=400;
w=0.1;
while maxIterJOR>0
    maxIterJOR=maxIterJOR-1;
    for i=1:M
        a_part_mu=(1-w).*q_mu_old(i,:);
        c_part_mu=zeros(1,N_newX);
        
        a_part_sigma=(1-w).*q_sigma_old(i,:);
        c_part_sigma=zeros(1,N_newX);
        for j=1:M
            %         for j=[m,Agents(m).Neighbors]
            K_A_i_j=squeeze(K_A(i,j,:))';
            if j>i
                c_part_mu=c_part_mu+K_A_i_j.*q_mu_old(j,:);
                c_part_sigma=c_part_sigma+K_A_i_j.*q_sigma_old(j,:);
            elseif j<i
                c_part_mu=c_part_mu+K_A_i_j.*q_mu_new(j,:);
                c_part_sigma=c_part_sigma+K_A_i_j.*q_sigma_new(j,:);
            else
                continue;
            end
            
        end
        K_A_i_i=squeeze(K_A(i,i,:))';
        b_part_mu=w./K_A_i_i.*(subMean(i,:)-c_part_mu);
        b_part_sigma=w./K_A_i_i.*(k_A(i,:)-c_part_sigma);
        
        q_mu_new(i,:)=a_part_mu+b_part_mu;
        q_sigma_new(i,:)=a_part_sigma+b_part_sigma;
    end
    q_mu_old=q_mu_new;
    q_sigma_old=q_sigma_new;
end
% calculation of q is not very accuarate
q_mu=q_mu_direct;
q_sigma=q_sigma_direct;

q_mu=q_mu_new;
q_sigma=q_sigma_new;

% JOR part end

%%%

direct_output_mu=zeros(1,N_newX);
% % % % % parfor n=1:N_newX
% % % % %     warning('off','all');
% % % % %     direct_output_mu(n)=k_A(:,n)'*inv(K_A(:,:,n))*subMean(:,n);
% % % % % end
%direct_output_mu=reshape(direct_output_mu,[64,64]);
% gcf=figure('visible','on');
% imshow(direct_output_mu,[]),colormap('jet');

% initial DAC
maxIterDAC=30;
% w_mu_old=k_A.*q_mu;
w_mu_old=k_A.*q_mu;
w_sigma_old=k_A.*q_sigma;
w_mu_new=w_mu_old;
w_sigma_new=w_sigma_old;
% DAC start (Debug Log 22 Apr: Funtional)
while maxIterDAC>0
    maxIterDAC=maxIterDAC-1;
    for m=1:M
        for n=1:M
            w_mu_new(m,:)=w_mu_new(m,:)+epsilon*(w_mu_old(n,:)-w_mu_old(m,:));
            w_sigma_new(m,:)=w_sigma_new(m,:)+epsilon*(w_sigma_old(n,:)-w_sigma_old(m,:));
        end
    end
    w_mu_old=w_mu_new;
    w_sigma_old=w_sigma_new;
end
w_mu=w_mu_new;
w_sigma=w_sigma_new;
mu=M*w_mu;
% var=(sigma_f.^2).*(k_star_star-M*w_sigma);
var=(k_star_star-M*w_sigma);
end

function [mu,var]=NN_NPAE2(Agents,newX,subMean,subVar,sigma_n)

epsilon=0.1;
[M,N_newX]=size(subMean);
k_A=zeros(M,N_newX);
K_A=zeros(M,M,N_newX);
C_mn=cell(M,M);
k_ms=cell(M,1);
q_mu_old=zeros(M,N_newX,M);
q_sigma_old=zeros(M,N_newX,M);
q_mu_new=zeros(M,N_newX);
q_sigma_new=zeros(M,N_newX);
k_star_star=zeros(M,N_newX);
sigma_f=zeros(M,1);
k_star_star=zeros(M,N_newX);

[Xs,Covs,invCovs]=getPartCovs(Agents);
for m=1:M
    theta=[Agents(m).sigma_f;Agents(m).l];
    k_ms{m}=getK(newX,Agents(m).X,theta,sigma_n); % N_newX * agent_N_data
end
for m=1:M
    %     K_A_m{m}=zeros(1,M,N_newX);
    K_A_m=zeros(M,N_newX);
    %     theta=[Agents(m).sigma_f;Agents(m).l];
    k_m=k_ms{m};
    %     k_A(m,:)=diag(k_m*invKs{m}*k_m');
    invKs_m=invCovs{m};
    for n=1:N_newX
        k_A(m,n)=k_m(n,:)*invKs_m*k_m(n,:)';
        k_star_star(m,n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
    end
    for n=1:M
        k_n=k_ms{n};
        
        for num=1:N_newX
            K_A_m(n,num)=k_m(num,:)*invKs_m*Covs{m,n}*invCovs{n}*k_n(num,:)';
            K_A(m,n,num)=k_m(num,:)*invKs_m*Covs{m,n}*invCovs{n}*k_n(num,:)';
        end
        
    end
    % pre initial JOR
    temp=ones(M,1)*subMean(m,:)./K_A_m(:,:);
    temp(isinf(temp))=0;
    q_mu_old(:,:,m)=temp;
    temp=k_A(:,:)./K_A_m(:,:);
    temp(isinf(temp))=0;
    q_sigma_old(:,:,m)=temp;
    %     K_A(m,:,:)=K_A_m;
end
% DALE begin

maxIterDALE=400;
P1=cell(M,1);
P2=cell(M,1);
ds=zeros(M,1);
q_mu_new=zeros(M,N_newX,M);
q_sigma_new=zeros(M,N_newX,M);
for m=1:M
    K_A_m=squeeze(K_A(m,:,:));
    P1{m}=K_A_m'./sum(K_A_m'.*K_A_m',2);
    P2{m}=zeros(M,M,N_newX);
    for n=1:N_newX
        P2{m}(:,:,n)=eye(M)-P1{m}(n,:)'*K_A_m(:,n)';
    end
    ds(m)=length(Agents(m).Neighbors);
end
q_mu=zeros(M,N_newX,M);
q_sigma=zeros(M,N_newX,M);
for iterCount=1:maxIterDALE
    for m=1:M
        P2_m=P2{m};
        neighbor=Agents(m).Neighbors;
        mu_sum_m=sum(q_mu_old(:,:,neighbor),3);
        sigma_sum_m=sum(q_sigma_old(:,:,neighbor),3);
        mu_part_2=zeros(M,N_newX);
        sigma_part_2=zeros(M,N_newX);
        for n=1:N_newX
            mu_part_2(:,n)=P2_m(:,:,n)*mu_sum_m(:,n);
            sigma_part_2(:,n)=P2_m(:,:,n)*sigma_sum_m(:,n);
        end
        q_mu_new_m=P1{m}'.*subMean(m,:)+1/ds(m)*mu_part_2;
        q_mu_new(:,:,m)=q_mu_new_m;
        q_sigma_new_m=P1{m}'.*subVar(m,:)+1/ds(m)*sigma_part_2;
        q_sigma_new(:,:,m)=q_sigma_new_m;
    end
    q_mu_old=q_mu_new;
    q_sigma_old=q_sigma_new;
end
q_mu=q_mu_new;
q_sigma=q_sigma_new;
mu=zeros(M,N_newX);
var=zeros(M,N_newX);
for m=1:M
    mu(m,:)=sum(k_A.*q_mu(:,:,m),1);
    var(m,:)=k_star_star-sum(k_A.*q_sigma(:,:,m),1);
end
end

function [Xs,Covs,invCovs]=getPartCovs(Agents)
M=length(Agents);
Xs=cell(1,M);
Covs=cell(M,M);
invCovs=cell(1,M);
for m=1:M
    Xs{m}=Agents(m).X;
end
for m=1:M
    theta=[Agents(m).sigma_f;Agents(m).l];
    neighbors=Agents(m).Neighbors;
    %     for n=[m,neighbors]
    for n=1:M
        Covs{m,n}=getK(Xs{m},Xs{n},theta,Agents(m).sigma_n);
    end
    invCovs{m}=inv(Covs{m,m});
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
% theta=Agent.z;
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
