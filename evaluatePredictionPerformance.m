function [pfmcMean,pfmcVar] = evaluatePredictionPerformance(realMean,realVar,preMean,preVar,method)
%EVALUATEPREDICTIONPERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
[M,N_newX]=size(preMean);
pfmcMean=[];
pfmcVar=[];
switch upper(method)
    case 'RMSE'
        errsquare_mean=zeros(M,N_newX);
        for m=1:M
            errsquare_mean(m,:)=(preMean(m,:)-realMean).^2;
        end
        pfmcMean=1/(M*N_newX)*sum(errsquare_mean(:));
        
        
        errsquare_var=zeros(M,N_newX);
        for m=1:M
            errsquare_var(m,:)=(preVar(m,:)-realVar).^2;
        end
        pfmcVar=1/(M*N_newX)*sum(errsquare_var(:));
    case 'NLPD'

end

end

