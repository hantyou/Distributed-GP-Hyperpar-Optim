function [sigma_pxADMM_fd,l_pxADMM_fd,outSteps,Zs] = runPXADMM_fd(Agents,M,epsilon,maxIter,sync)
%RUNPXADMM_FD Summary of this function goes here
%   Detailed explanation goes here
if sync
    inputDim=size(Agents(1).X,1);
    
    sampleSize=M*length(Agents(1).Z);
    pxADMM_fd_flag=1;
    Sigmas=[];
    Sigmas=[Sigmas;Agents(1).sigma_f];
    Ls=[];
    Ls=[Ls,Agents(1).l];
    Likelihood=[];
    iterCount=0;
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
    SubSteps=cell(M,1);
    for m=1:M
        SubSteps{m}=[];
    end
    Steps=cell(M,1);
    for m=1:M
        Steps{m}=[];
    end
    outSteps=[];
    %wb=waitbar(0,'Preparing','Name','pxADMM_{fd}');
    %set(wb,'color','w');
    sub_old_z=zeros(inputDim+1,M);
    for m=1:M
        sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
    end
    while pxADMM_fd_flag>0
        iterCount=iterCount+1;
        
        if mod(iterCount,250)==0
           disp(iterCount); 
        end
        
        Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
        old_z=updated_z;
        updated_z=zeros(inputDim+1,1);
        global_step=0;
        
        
        for m=1:M
            Agents(m)=Agents(m).runPxADMM_fd;
            Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l]];
            step_m=max(vecnorm([Agents(m).sigma_f;Agents(m).l]-sub_old_z(:,m),2,2));
%             step_m=max(vecnorm(Agents(m).l-sub_old_z(2:end,m),2,2));
            SubSteps{m}=[SubSteps{m},step_m];
            global_step=max(global_step,step_m);
        end
        for m=1:M
            Agents(m).action_status=1;
            updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l];
            
            sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
            %     Agents(m).rho=Agents(m).rho*0.9999;
        end
        updated_z=updated_z/M;
        step = max(vecnorm(updated_z(1:end)-old_z(1:end),2,2));
        outSteps=[outSteps,step];
        
        %waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM_{fd}: ',iterCount/maxIter*100,'% , step:', step))
        
        if step < epsilon
            pxADMM_fd_flag=0;
        end
        if iterCount>=maxIter
            pxADMM_fd_flag=0;
        end
    end
    sigma_pxADMM_fd=updated_z(1);
    l_pxADMM_fd=updated_z(2:end);
    
    gcf=figure,
    cvgValue=Inf*ones(1,inputDim+1);
    for i=1:inputDim+1
        subplot(inputDim+1,1,i)
        if i==1
            ylabel('s_f')
        else
            txt=strcat('l_',num2str(i-1));
            ylabel(txt)
        end
        hold on
        for m=1:M
            plot(Zs{m}(i,:));
            cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
        end
        lgd=yline(cvgValue(i),'-.b');
        if i==1
            legend(lgd,'converged value', 'Location','northwest')
        end
        set(gca, 'XScale', 'log')
        hold off
        
        %     set(gca,'XScale','log')
    end
    fname='results/pxADMM_fd_sync_cvg_value_direct_display';
    saveas(gcf,fname,'png');
    close gcf;
    
    outSteps=SubSteps;
    
    gcf=figure,
    hold on;
    for m=1:M
        plot(outSteps{m})
    end
    legendText=[];
    for m=1:M
        if m<10
            legendText=[legendText;strcat('agent-0',num2str(m))];
        else
            legendText=[legendText;strcat('agent-',num2str(m))];
        end
    end
    legend(legendText);
    xlabel('steps')
    ylabel('step length')
    title('pxADMMfd convergence')
    set(gca,'YScale','log')
    hold off;
    fname='results/pxADMM_fd_sync_cvg_value_steps';
    saveas(gcf,fname,'png');
    close gcf;

    %delete(wb);
else
    inputDim=size(Agents(1).X,1);
    sampleSize=M*length(Agents(1).Z);
    pxADMM_fd_flag=1;
    Sigmas=[];
    Sigmas=[Sigmas;Agents(1).sigma_f];
    Ls=[];
    Ls=[Ls,Agents(1).l];
    Likelihood=[];
    iterCount=0;
    % the outer iteration begains here
    updated_z=[Agents(1).sigma_f;Agents(1).l];
    Zs=cell(M,1);
    for m=1:M
        Zs{m}=[];
    end
    SubSteps=cell(M,1);
    for m=1:M
        SubSteps{m}=[];
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
    stepForStops=[];
%     wb=waitbar(0,'Preparing','Name','pxADMM_{fd}');
%     set(wb,'color','w');
    sub_old_z=zeros(inputDim+1,M);
    for m=1:M
        sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
    end
    p01=0.5;
    p10=0.9;
    
    P01s=rand(1,M)*0.6+0.4;
    P10s=rand(1,M)*0.4+0.6;
    
    AgentsActionStatus=[];
    while pxADMM_fd_flag
        % Randomly activate some of the agents
        activatedAgents=zeros(M,1);
        activated_agents=[];
        for m=1:M
            if Agents(m).action_status||Agents(m).communicationAbility
                %                 if rand<p10
                %                     Agents(m).action_status=0;
                %                 else
                %                     Agents(m).action_status=1;
                %                     activatedAgents(m)=1;
                %                     activated_agents=[activated_agents,m];
                %                 end
                
                Agents(m).updatedVarsNumber=sum(Agents(m).updatedVars);
                if Agents(m).updatedVarsNumber>=Agents(m).updatedVarsNumberThreshold
                    Agents(m).communicationAbility=0;
                    Agents(m).action_status=1;
                    activated_agents=[activated_agents,m];
                    activatedAgents(m)=1;
                else
                    Agents(m).communicationAbility=1;
                    Agents(m).action_status=0;
                end
            elseif  Agents(m).action_status==0&&Agents(m).communicationAbility==0
                if rand<P01s(m)
                    Agents(m).action_status=1;
                    activatedAgents(m)=1;
                    Agents(m).communicationAbility=1;
                    activated_agents=[activated_agents,m];
                else
                    Agents(m).action_status=0;
                    Agents(m).communicationAbility=0;
                end
            end
        end
        if isempty(activated_agents)
            Agents(1).action_status=1;
            activatedAgents(1)=1;
            activated_agents=1;
        end
        AgentsActionStatus=[AgentsActionStatus,activatedAgents];
        % % % %         N_activated_agents=unidrnd(M,1);
        % % % %         randsequence=randperm(M);
        % % % %         activated_agents=randsequence(1:N_activated_agents);
        % % % %         non_activated_agents=randsequence(N_activated_agents:end);
        % % % %
        % % % %         if ~isempty(non_activated_agents)
        % % % %             non_activated_agents(1)=[];
        % % % %         end
        % % % %         for m_a=activated_agents
        % % % %             Agents(m_a).action_status=1;
        % % % %         end
        % % % %         for m_n=non_activated_agents
        % % % %             Agents(m_n).action_status=0;
        % % % %         end
        % for those agents who are activated, obtain variables from their
        % activated neighbours
        iterCount=iterCount+1;
        Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
        old_z=updated_z;
        updated_z=zeros(inputDim+1,1);
        global_step=0;
        
        
        for i=1:length(activated_agents)
            m=activated_agents(i);
            Agents(m)=Agents(m).runPxADMM_fd;
            Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l]];
            step_m=max(vecnorm([Agents(m).sigma_f;Agents(m).l]-sub_old_z(:,m),2,2));
%             step_m=max(vecnorm(Agents(m).l-sub_old_z(2:end,m),2,2));
            SubSteps{m}=[SubSteps{m},step_m];
            global_step=max(global_step,step_m);
        end
        for m=1:M
            updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l];
            
            sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l];
            %     Agents(m).rho=Agents(m).rho*0.9999;
        end
        updated_z=updated_z/M;
        step = max(vecnorm(global_step,2,2));
        %         step = max(vecnorm(updated_z-old_z,2,2));
        outSteps=[outSteps,step];
        if iterCount>5
            stepForStop=mean(outSteps(end-5:end));
        else
            stepForStop=mean(outSteps(1:end));
        end
        stepForStops=[stepForStops;stepForStop];
        
%         waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM_{fd,async}: ',iterCount/maxIter*100,'% , step:', step))
        
        if stepForStop < epsilon
            pxADMM_fd_flag=0;
        end
        if iterCount>=maxIter
            pxADMM_fd_flag=0;
        end
    end
    outSteps=stepForStops;
    sigma_pxADMM_fd=updated_z(1);
    l_pxADMM_fd=updated_z(2:end);
    % Plot
    %     figure,
    %     for m=1:M
    %         for z_i=1:3
    %             subplot(3,M,(z_i-1)*M+m);
    %             plot(Zs{m}(z_i,:));
    %         end
    %     end
    
    gcf=figure,
    cvgValue=Inf*ones(1,inputDim+1);
    for i=1:inputDim+1
        subplot(inputDim+1,1,i)
        if i==1
            ylabel('s_f')
        else
            txt=strcat('l_',num2str(i-1));
            ylabel(txt)
        end
        hold on
        if i==1
        for m=1:M
            plot(Zs{m}(i,:));
            cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
            %ylim([4 6.2])
        end
        else
        for m=1:M
            plot(Zs{m}(i,:));
            cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
            %ylim([-2 2])
        end
        end
        lgd=yline(cvgValue(i),'-.b');
        if i==1
            legend(lgd,'converged value', 'Location','northwest')
        end
        set(gca, 'XScale', 'log')
        hold off
        
        %     set(gca,'XScale','log')
    end
    fname='results/pxADMM_fd_async_cvg_values_direct_display'
    saveas(gcf,fname,'png');
    close gcf;
    
    figure,
    for m=1:M
        subplot(M,1,m);
        status=kron(AgentsActionStatus(m,1:50),ones(1,10));
        area(linspace(0,51,length(status)),status);
    end
    
    
    %     figure,
    %     for m=1:M
    %         subplot(2,ceil(M/2),m);
    %         semilogy(SubSteps{m});
    %     end
    %     outSteps=SubSteps{1};
    outSteps=SubSteps;
    
    gcf=figure,
    hold on;
    for m=1:M
        plot(outSteps{m})
    end
    legendText=[];
    for m=1:M
        if m<10
            legendText=[legendText;strcat('agent-0',num2str(m))];
        else
            legendText=[legendText;strcat('agent-',num2str(m))];
        end
    end
    legend(legendText);
    xlabel('steps')
    ylabel('step length')
    title('pxADMMfd convergence')
    set(gca,'YScale','log')
    hold off;
    
    fname='results/pxADMM_fd_async_steps_direct_display'
    saveas(gcf,fname,'png');
    close gcf;
%     delete(wb);
end
end

