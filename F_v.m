function y= F_v(y,Q_NN,K_tilde,sigma)
%F_V Summary of this function goes here
%   Detailed explanation goes here
[~,N]=size(y);
y=y';
Sigma=sigma^2*eye(N)+Q_NN;
part_1=-0.5*log(det(2*pi*Sigma))-0.5*y'/Sigma*y;
% part_2=-0.5/sigma^2*trace(K_tilde);
part_2=-0.5/1^2*trace(K_tilde);
y=part_1+part_2;
end

