function [subMeans,subVars] = GPR_predict_NoAg(Agents,newX,sigma_n)
%GPR_PREDICT_DEC Summary of this function goes here
%   Detailed explanation goes here
if nargin==2
    sigma_n=0;
end
disp('No-Agg')
M=length(Agents);
N_newX=size(newX,2);

subMeans=zeros(M,N_newX);
subVars=zeros(M,N_newX);

for m=1:M
    [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
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