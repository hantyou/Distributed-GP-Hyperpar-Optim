function [pfmcMean,pfmcVar] = evaluatePredictionPerformance(realMean,realVar,preMean,preVar,method)
%EVALUATEPREDICTIONPERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
[M,N_newX]=size(preMean);
switch upper(method)
    case 'RMSE'
        
    case 'NLPD'

end

end

