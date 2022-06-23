function [pd,pdn] = getDiv(obj,z)
%GETDIV Summary of this function goes here
%   Detailed explanation goes here
D=length(obj.l);
sigma_old=z(1);
l_old=z(2:(2+D-1));
inputDim=length(l_old);
K_n=obj.sigma_n^2*eye(obj.N_m);
%%%%local update START%%%%
distX = dist((diag(l_old)\eye(inputDim))*obj.X).^2;%distX=(X-X^')^T Sigma^-1(X-X^')
%             K_s=obj.sigma_f^(2)*exp(-0.5*distX./obj.l^(2));
K_s=sigma_old^(2)*exp(-0.5*distX);
obj.K=K_s+K_n;

choL = chol(obj.K, 'lower');
alpha = choL'\(choL\obj.Z);
invChoL=inv(choL);
constant_1=invChoL'*invChoL-alpha*alpha';
%% div sigma_f
K_div_sigma_f=2/sigma_old*K_s;
pd_sigma_f = 0.5*trace(constant_1*K_div_sigma_f);
%% div l
pd_l=zeros(inputDim,1);
for d=1:inputDim
    K_div_l_d=obj.distXd(:,:,d).*K_s*l_old(d)^(-3);
    pd_l(d) = 0.5*trace(constant_1*K_div_l_d);
end
%% div sigma_n
if nargout>1
    K_div_sigma_n=2*sqrt(K_n);
    pdn=0.5*trace(constant_1*K_div_sigma_n);
end


pd=[pd_sigma_f;pd_l];
end

