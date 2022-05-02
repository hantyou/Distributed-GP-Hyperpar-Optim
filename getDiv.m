function pd = getDiv(obj,z)
%GETDIV Summary of this function goes here
%   Detailed explanation goes here
sigma_old=z(1);
l_old=z(2:end);
inputDim=length(l_old);
K_n=obj.sigma_n^2*eye(obj.N_m);
%%%%local update START%%%%
distX = dist((diag(l_old)\eye(inputDim))*obj.X).^2;%distX=(X-X^')^T Sigma^-1(X-X^')
%             K_s=obj.sigma_f^(2)*exp(-0.5*distX./obj.l^(2));
K_s=sigma_old^(2)*exp(-0.5*distX);
obj.K=K_s+K_n;

choL = chol(obj.K, 'lower');
alpha = choL'\(choL\obj.Z);
%             invK=inv(obj.K);
invChoL=inv(choL);
constant_1=invChoL'*invChoL-alpha*alpha';
K_div_sigma_f=2/sigma_old*K_s;
%             K_div_sigma_f=2*obj.sigma_f*exp(-0.5*distX/obj.l^2);
pd_sigma_f = 0.5*trace(constant_1*K_div_sigma_f);

%             K_div_l=obj.sigma_f^2*distX*exp(-distX./2./obj.l^(2))*obj.l^(-3);
pd_l=zeros(inputDim,1);
for d=1:inputDim
    K_div_l_d=obj.distXd(:,:,d).*K_s*l_old(d)^(-3);
    pd_l(d) = 0.5*trace(constant_1*K_div_l_d);
end
% K_div_l_1=obj.distX1.*K_s*l_old(1)^(-3);
% pd_l_1 = 0.5*trace(constant_1*K_div_l_1);
% 
% K_div_l_2=obj.distX2.*K_s*l_old(2)^(-3);
% pd_l_2 = 0.5*trace(constant_1*K_div_l_2);
% 
% K_div_l=[pd_l_1;pd_l_2];
% pd_l=K_div_l;

pd=[pd_sigma_f;pd_l];
end

