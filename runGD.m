function [newSigma,newL,Steps] = runGD(Agents,M,initial_sigma_f,initial_l,sigma_n,stepSize,stopCriteria,maxIter)
%RUNGD Summary of this function goes here
%   Detailed explanation goes here
sampleSize=M*length(Agents(1).Z);
epsilon=stopCriteria;
GDflag=1;
Sigmas=zeros(maxIter,1);
Sigmas(1)=initial_sigma_f;
Sigma_ns=zeros(maxIter,1);
Sigma_ns(1)=0.1;
Ls=zeros(2,maxIter);
Ls(:,1)=initial_l;
Steps=zeros(maxIter,1);
Likelihoods=zeros(maxIter,1);
iterCount=1;
Zs=cell(M,1);
for m=1:M
    Zs{m}=[];
end
for m=1:M
    Agents(m) = Agents(m).initLocalGD(initial_sigma_f,initial_l',stepSize);
    Agents(m).sigma_n=sigma_n;
    Agents(m).pd_l=[0;0];
end
% perform GD
% wb=waitbar(0,'Preparing','Name','GD');
% set(wb,'color','w');
while GDflag
    iterCount=iterCount+1;
    oldSigma=Sigmas(iterCount-1);
    oldL=Ls(:,iterCount-1);
    newSigma=Sigmas(iterCount-1);
    newL=Ls(:,iterCount-1);
%     newSigma_n=Sigma_ns(iterCount-1);
    newLikelihood=0;
%     spmd(M)
%         obj=Agents(labindex);
%         obj = obj.runLocalGD;
%     end
%     Agents=[obj{:}];
    for m=1:M
        Agents(m) = Agents(m).runLocalGD;
        newSigma=newSigma-2*Agents(m).mu*Agents(m).pd_sigma_f;
        newL=newL-Agents(m).mu*Agents(m).pd_l;
%         newSigma_n=newSigma_n-Agents(m).mu*Agents(m).pd_sigma_n;
        newLikelihood=newLikelihood+Agents(m).Z'*inv(Agents(m).K)*Agents(m).Z+log(det(Agents(m).K));
    end
    
    Sigmas(iterCount)=newSigma;
    Ls(:,iterCount)=newL;
    Likelihoods(iterCount)=-0.5*newLikelihood+0.5*sampleSize*log(2*pi);
    for m=1:M
        Agents(m).sigma_f=newSigma;
        Agents(m).l=newL;
        Zs{m}=[Zs{m},[newSigma;newL]];
%         Agents(m).sigma_n=newSigma_n;
    end
%         step=max(abs(Sigmas(iterCount)-Sigmas(iterCount-1)),abs(Ls(1,iterCount)-Ls(1,iterCount-1)));
    step=max(vecnorm([newSigma;newL]-[oldSigma;oldL],2,2))  ;  
    Steps(iterCount)=step;
    
%     waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','GD: ',iterCount/maxIter*100,'% , step:', step))
    if step<epsilon
        GDflag=0;
    end
    if iterCount>=maxIter
        GDflag=0;
    end
%     if ~mod(iterCount,100)
%         disp(iterCount)
%         disp(newLikelihood)
%     end
end
% Plot
figure,
for m=1:M
    for z_i=1:3
        subplot(3,M,(z_i-1)*M+m);
        plot(Zs{m}(z_i,:));
    end
end
figure,semilogy(Steps)
xlabel('steps')
ylabel('step length')
title('GD convergence')
% delete(wb);
end

