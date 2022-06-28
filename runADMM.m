function [sigma_ADMM,l_ADMM,sigma_n_ADMM,Steps,IterCounts] = runADMM(Agents,M,stepSize,epsilon,maxOutIter,maxInIter)
%RUNADMM Summary of this function goes here
%   Detailed explanation goes here
sampleSize=M*length(Agents(1).Z);
outADMMflag=1;
Sigmas=[];
Sigmas=[Sigmas;Agents(1).sigma_f];
Ls=[];
Ls=[Ls,Agents(1).l];
Likelihood=[];
outIterCount=0;
% the outer iteration begains here
updated_z=Agents(1).z;
Zs=cell(M,1);
for m=1:M
    Zs{m}=[];
end
IterCounts=cell(M,1);
for m=1:M
    IterCounts{m}=0;
end
Steps=cell(M,1);
for m=1:M
    Steps{m}=[];
end
outSteps=[];
wb=waitbar(0,'Preparing','Name','ADMM');
set(wb,'color','w');
rho_0=Agents(1).rho;
maxInIter_0=maxInIter;
while outADMMflag
    outIterCount=outIterCount+1;
    %     disp(outIterCount)
    % Calculate z from agents' data
    old_z=updated_z;
    updated_z=zeros(size(updated_z));
    for m=1:M
        Agents(m).mu=stepSize;
        updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]+1/Agents(m).rho*Agents(m).beta;
    end
    updated_z=1/M*updated_z;
    % transmit z from center to agents
    for m=1:M
        Agents(m).z=updated_z;
    end
    % at each agent, perform inner iteration
    % for each inner iteration, the theta=[sigma_f, l] is going to be
    % updated, and beta is also to be updated
    for m=1:M
        [Agents(m),Zs_m,Steps_m,inIterCount_m]=Agents(m).runLocalInnerADMM(maxInIter,epsilon);
        Zs{m}=[Zs{m},Zs_m];
        Steps{m}=[Steps{m},Steps_m];
        IterCounts{m}=[IterCounts{m},IterCounts{m}(outIterCount)+inIterCount_m];
    end
    
    step = norm(updated_z-old_z);
    outSteps=[outSteps,step];
    
    waitbar(outIterCount/maxOutIter,wb,sprintf('%s %.2f %s %f','ADMM: ',outIterCount/maxOutIter*100,'% , step:', step))
    if step < epsilon
        outADMMflag=0;
    end
    if outIterCount>=maxOutIter
        outADMMflag=0;
    end
end
sigma_ADMM=updated_z(1);
D=length(Agents(1).l);
l_ADMM=updated_z(2:(2+D-1));
sigma_n_ADMM=updated_z(end);
% Plot
figure,
for m=1:M
    for z_i=1:3
        subplot(3,M,(z_i-1)*M+m);
        plot(Zs{m}(z_i,:));
    end
    IterCounts{m}(1)=[];
end
figure,semilogy(IterCounts{1},outSteps);
Steps=outSteps;
delete(wb);

end

