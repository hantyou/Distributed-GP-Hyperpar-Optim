% NPAE branch
function [Mean,Var,mean,var] = GPR_predict_dec(Agents,method,newX,A,maxIter,sigma_n)
%GPR_PREDICT_DEC Summary of this function goes here
%   Detailed explanation goes here
if nargin==5
    sigma_n=0;
end

M=length(Agents);
N_newX=size(newX,2);

subMeans=zeros(M,N_newX);
subVars=zeros(M,N_newX);

parfor m=1:M
    [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
end
maxIter=20;

switch upper(method)
    case {'DEC-POE'}
        disp('DEC-PoE')
        %%%
        beta=ones(M,1);
        [Mean,Var,mean,var]=DEC_gPoE(A,subMeans,subVars,beta,maxIter);
        %%%
        %         disp('Function Under Construction')
    case {'DEC-GPOE'}
        disp('DEC-gPoE')
        %%%
        beta=ones(M,1)/M;
        [Mean,Var,mean,var]=DEC_gPoE(A,subMeans,subVars,beta,maxIter);
        %%%
        %         disp('Function Under Construction')
    case {'DEC-BCM'}
        disp('DEC-BCM')
        %%%
        k_star_star=zeros(M,N_newX);
        for m=1:M
            theta=[Agents(m).sigma_f;Agents(m).l];
            for n=1:N_newX
                k_star_star(m,n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
            end
        end
        beta=ones(M,N_newX);
        [Mean,Var,mean,var]=DEC_rBCM(A,subMeans,subVars,beta,k_star_star,maxIter);

        %%%
        %         disp('Function Under Construction')
    case {'DEC-RBCM'}
        disp('DEC-rBCM')
        %%%
        k_star_star=zeros(M,N_newX);
        for m=1:M
            theta=[Agents(m).sigma_f;Agents(m).l];
            for n=1:N_newX
                k_star_star(m,n)=kernelFunc(newX(:,n),newX(:,n),theta,sigma_n,'RBF');
            end
        end
        %         beta=ones(M,N_newX);
        beta=0.5*(log(k_star_star)-log(subVars));
        [Mean,Var,mean,var]=DEC_rBCM(A,subMeans,subVars,beta,k_star_star,maxIter);

        %%%
        %         disp('Function Under Construction')
    case {'DEC-GRBCM'}
        disp('DEC-grBCM')
        %%%

        %%%
        disp('Function Under Construction')
    case {'DEC-NPAE'}
        disp('DEC-NPAE')
        %%%
        [mean,var]=DEC_NPAE(Agents,newX,subMeans,subVars,maxIter,sigma_n);
        Mean=0;
        Var=0;
        %%%
%         disp('Function Under Construction')

end

end

%% Functions

function [Means,inv_Vars,mu,var]=DEC_gPoE(A,subMean,subVar,beta,maxIter)
epsilon=0.1;
[M,N_newX]=size(subMean);
Means=zeros(M,N_newX,maxIter+1);
inv_Vars=zeros(M,N_newX,maxIter+1);
Means(:,:,1)=beta.*(1./subVar).*subMean;
inv_Vars(:,:,1)=beta.*(1./subVar);
for i=1:maxIter
    for m=1:M
        % Var
        inv_var_m_i=inv_Vars(m,:,i);
        inv_var_m_j_i=zeros(1,N_newX);
        for n=1:M
            inv_var_m_j_i=inv_var_m_j_i+A(m,n)*(inv_Vars(n,:,i)-inv_Vars(m,:,i));
        end
        inv_var_m_ip1=inv_var_m_i+epsilon*inv_var_m_j_i;
        % Mean
        mu_m_i=Means(m,:,i);
        mu_m_j_i=zeros(1,N_newX);
        for n=1:M
            mu_m_j_i=mu_m_j_i+A(m,n)*(Means(n,:,i)-Means(m,:,i));
        end
        mu_m_ip1=mu_m_i+epsilon*mu_m_j_i;
        % Store
        Means(m,:,i+1)=mu_m_ip1;
        inv_Vars(m,:,i+1)=inv_var_m_ip1;
    end
end

if nargout>2
    var=1./(M*inv_Vars(:,:,end));
    mu=M*var.*Means(:,:,end);
end
end

%%%

function [Means,inv_Vars,mu,var]=DEC_rBCM(A,subMean,subVar,beta,k_star_star,maxIter)
epsilon=0.1;
[M,N_newX]=size(subMean);
Means=zeros(M,N_newX,maxIter+1);
inv_Vars=zeros(M,N_newX,maxIter+1);
Means(:,:,1)=beta.*(1./subVar).*subMean;
inv_Vars(:,:,1)=beta.*(1./subVar);
for i=1:maxIter
    for m=1:M
        % Var
        inv_var_m_i=inv_Vars(m,:,i);
        inv_var_m_j_i=zeros(1,N_newX);
        for n=1:M
            inv_var_m_j_i=inv_var_m_j_i+A(m,n)*(inv_Vars(n,:,i)-inv_Vars(m,:,i));
        end
        inv_var_m_ip1=inv_var_m_i+epsilon*inv_var_m_j_i;
        % Mean
        mu_m_i=Means(m,:,i);
        mu_m_j_i=zeros(1,N_newX);
        for n=1:M
            mu_m_j_i=mu_m_j_i+A(m,n)*(Means(n,:,i)-Means(m,:,i));
        end
        mu_m_ip1=mu_m_i+epsilon*mu_m_j_i;
        % Store
        Means(m,:,i+1)=mu_m_ip1;
        inv_Vars(m,:,i+1)=inv_var_m_ip1;
    end
end

if nargout>2
    var=1./(M*inv_Vars(:,:,end)+(1-sum(beta,1))./k_star_star);
    mu=M*var.*Means(:,:,end);
end
end

%%%
%% NPAE function
function [mu,var]=DEC_NPAE(Agents,newX,subMean,subVar,maxIter,sigma_n)
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

parfor m=1:M
    sigma_f(m)=Agents(m).sigma_f;
    X1=Agents(m).X;
    theta=[Agents(m).sigma_f;Agents(m).l];
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

parfor m=1:M
%     K_A_m{m}=zeros(1,M,N_newX);
    K_A_m=zeros(M,N_newX);
%     theta=[Agents(m).sigma_f;Agents(m).l];
    k_m=k_ms{m};
    k_A(m,:)=diag(k_m*invKs{m}*k_m');
    for n=1:M
        k_n=k_ms{n};
        K_A_m(n,:)=diag(k_m*invKs{m}*C_mn{m,n}*invKs{n}'*k_n')';
    end
    % pre initial JOR
    q_mu_old(m,:)=subMean(m,:)./K_A_m(m,:);
    q_sigma_old(m,:)=k_A(m,:)./K_A_m(m,:);
    K_A(m,:,:)=K_A_m;
end
% JOR initial
q_mu_direct=zeros(M,N_newX);
q_sigma_direct=zeros(M,N_newX);
parfor n=1:N_newX
    q_mu_direct(:,n)=inv(K_A(:,:,n))*subMean(:,n);
    q_sigma_direct(:,n)=inv(K_A(:,:,n))*k_A(:,n);
end
maxIterJOR=200;
w=0.2;
while maxIterJOR>0
    maxIterJOR=maxIterJOR-1;
    for i=1:M
        a_part_mu=(1-w).*q_mu_old(i,:);
        c_part_mu=zeros(1,N_newX);

        a_part_sigma=(1-w).*q_sigma_old(i,:);
        c_part_sigma=zeros(1,N_newX);
        for j=1:M
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
% JOR part end

%%%

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
var=sigma_f.*(k_star_star-M*w_sigma);

end

%%%
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

for i=1:M
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
