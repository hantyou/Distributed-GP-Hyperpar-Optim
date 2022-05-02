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

for m=1:M
    [subMeans(m,:),subVars(m,:)]=subGP(Agents(m),newX,sigma_n);
end


switch upper(method)
    case {'NN-POE'}
    case {'NN-GPOE'}
    case {'NN-BCM'}
    case {'NN-GRBCM'}
    case {'NN-NPAE'}
end

end
