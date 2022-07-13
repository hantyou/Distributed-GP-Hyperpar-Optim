function [sigma_pxADMM,l_pxADMM,sigma_n_pxADMM,Steps,Zs] = runPXADMM(Agents,M,epsilon,maxIter)
%RUNPXADMM Summary of this function goes here
%   Detailed explanation goes here
sampleSize=M*length(Agents(1).Z);
pxADMMflag=1;
Sigmas=[];
Sigmas=[Sigmas;Agents(1).sigma_f];
Ls=[];
Ls=[Ls,Agents(1).l];
Steps=[];
Likelihood=[];
iterCount=0;
% the outer iteration begains here
updated_z=Agents(1).z;
Zs=cell(M,1);
for m=1:M
    Zs{m}=[];
end
D=length(Agents(1).l);
%wb=waitbar(0,'Preparing','Name','pxADMM');
%set(wb,'color','w');
while pxADMMflag
    iterCount=iterCount+1;
    % Calculate z from agents' data
    old_z=updated_z;
    inputDim=size(Agents(1).X,1);
    updated_z=zeros(inputDim+2,1);
    for m=1:M
        updated_z=updated_z+...
            [Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]...
            +1/Agents(m).rho*Agents(m).beta;
    end
    updated_z=1/M*updated_z;
    % transmit z from center to agents
    for m=1:M
        Agents(m).z=updated_z;
    end
    % at each agent, perform one step of proximal update, get the new
    % hyperparameters
    for m=1:M
        Agents(m)=Agents(m).runLocalPxADMM;
        Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]];
    end
    step = norm(updated_z(1:end)-old_z(1:end));
    Steps=[Steps,step];
    
%    waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM: ', iterCount/maxIter*100,'% , step:', step))
    if step < epsilon
        pxADMMflag=0;
    end
    if iterCount>=maxIter
        pxADMMflag=0;
    end
end
sigma_pxADMM=updated_z(1);
l_pxADMM=updated_z(2:(2+D-1));
sigma_n_pxADMM=updated_z(end);
% plot part
% figure,
% for m=1:M
%     for z_i=1:3
%         subplot(3,M,(z_i-1)*M+m);
%         plot(Zs{m}(z_i,:));
%     end
% end
gcf=figure;
cvgValue=Inf*ones(1,inputDim+1);
for i=1:inputDim+1
    subplot(inputDim+1,1,i)
    hold on
    for m=1:1
        plot(Zs{m}(i,:));
        cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
    end
        yline(cvgValue(i),'-.r');
    hold off
end
figure,semilogy(Steps)
xlabel('steps')
ylabel('step length')
title('pxADMM convergence')
fname='pxADMM convergence';
saveas(gcf,fname,'png')
close gcf;


%delete(wb);

end

