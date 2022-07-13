function [Mean,Var] = GPRSinglePointPredict(X,x,alpha,L,theta,sigma_n)
%GPRSINGLEPOINTPREDICT Summary of this function goes here
%   Detailed explanation goes here

% unpack variables
N=size(X,2);
sigma_f=theta(1);
l=theta(2:(end-1));


x_star=x;
K_star_star=k(x_star,x_star,sigma_f,l,sigma_n);
K_star=zeros(1,N);
for i=1:N
    K_star(i)=k(x_star,X(:,i),sigma_f,l,sigma_n);
end
Mean=K_star*alpha;
%     y_est=K_star*inv(K)*y;
v=L\K_star';
Var=K_star_star-v'*v;
end

