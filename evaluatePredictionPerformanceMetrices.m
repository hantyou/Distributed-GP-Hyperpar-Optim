function [pfmcMean,pfmcVar] = evaluatePredictionPerformanceMetrices(realMean,realVar,preMean,preVar,method)
%EVALUATEPREDICTIONPERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
[M,N_newX,iterNum]=size(preMean);
pfmcMean=[];
pfmcVar=[];
switch upper(method)
    case 'RMSE'
        for i=1:iterNum
            errsquare_mean=zeros(M,N_newX);
            for m=1:M
                errsquare_mean(m,:)=(preMean(m,:,i)-realMean).^2;
            end
            pfmcMean=[pfmcMean;1/(M*N_newX)*sum(errsquare_mean(:))];


            errsquare_var=zeros(M,N_newX);
            for m=1:M
                errsquare_var(m,:)=(preVar(m,:,i)-realVar).^2;
            end
            pfmcVar=[pfmcVar;1/(M*N_newX)*sum(errsquare_var(:))];
        end
    case 'NLPD'

    case upper('consensusRMSE')
        realMean_avg=sum(preMean(:,:,1),1)/M;
        realVar_avg=sum(preVar(:,:,1),1)/M;
        for i=1:iterNum
            errsquare_mean=zeros(M,N_newX);
            for m=1:M
                errsquare_mean(m,:)=(preMean(m,:,i)-realMean_avg).^2;
            end
            pfmcMean=[pfmcMean;1/(M*N_newX)*sum(errsquare_mean(:))];


            errsquare_var=zeros(M,N_newX);
            for m=1:M
                errsquare_var(m,:)=(preVar(m,:,i)-realVar_avg).^2;
            end
            pfmcVar=[pfmcVar;1/(M*N_newX)*sum(errsquare_var(:))];
        end

end

end

