function K=getK(X1,X2,theta,sigma_n)
if nargin<4
    sigma_n=theta;
    theta=X2;

    X=X1;
    X2=X1;
end


[D,M]=size(X1);
[~,N]=size(X2);
sigma_f=theta(1);
l=theta(2:(2+D-1));
K=zeros(M,N);

for i=1:M
    for j=1:N
        K(i,j)=kernelFunc(X1(:,i),X2(:,j),theta,sigma_n,'RBF');
    end
end
end