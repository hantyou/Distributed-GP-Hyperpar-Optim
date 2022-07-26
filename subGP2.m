function [Mean,Var]=subGP2(X,Z,newX,theta,sigma_n)
if nargin==2
    sigma_n=0;
end

N_newX=size(newX,2);

[m,n]=size(Z);
if m<n
    Z=Z';
end

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