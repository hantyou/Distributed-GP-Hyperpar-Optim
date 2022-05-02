function [sigma_ADMM_fd,l_ADMM_fd,Steps,IterCounts] = runADMM_fd(Agents,M,stepSize,epsilon,maxOutIter,maxInIter)
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
updated_z=[Agents(1).sigma_f;Agents(1).l];
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
%wb=waitbar(0,'Preparing','Name','ADMM_{fd}');
%set(wb,'color','w');
rho_0=Agents(1).rho;
maxInIter_0=maxInIter;
sub_old_z=zeros(3,M);
for m=1:M
    sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
end
while outADMMflag
    outIterCount=outIterCount+1;
    Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
    global_step=0;
    updated_z=0;
    for m=1:M
        Agents(m)=Agents(m).runLocalInnerADMM_fd(maxInIter,epsilon);
        [Agents(m),Zs_m,Steps_m,inIterCount_m]=Agents(m).runLocalInnerADMM_fd(maxInIter,epsilon);
        Zs{m}=[Zs{m},Zs_m];
        Steps{m}=[Steps{m},Steps_m];
        IterCounts{m}=[IterCounts{m},IterCounts{m}(outIterCount)+inIterCount_m];
        step_m=norm([Agents(m).sigma_f;Agents(m).l]-sub_old_z(:,m));
        global_step=max(global_step,step_m);
        sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
        updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l];
    end
    updated_z=updated_z/M;
    step = global_step;
    outSteps=[outSteps,step];
    
%    waitbar(outIterCount/maxOutIter,wb,sprintf('%s %.2f %s %f','ADMM_{fd}: ',outIterCount/maxOutIter*100,'% , step:', step))
    if step < epsilon
        outADMMflag=0;
    end
    if outIterCount>=maxOutIter
        outADMMflag=0;
    end
end
sigma_ADMM_fd=updated_z(1);
l_ADMM_fd=updated_z(2:3);
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
%delete(wb);

end

