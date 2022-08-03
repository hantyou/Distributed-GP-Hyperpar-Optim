function [sigma_pxADMM_fd,l_pxADMM_fd,sigma_n_pxADMM_fd,outSteps,Zs,thetas] = runPXADMM_fd(Agents,M,epsilon,maxIter,sync)
%RUNPXADMM_FD Summary of this function goes here
%   Detailed explanation goes here
D=length(Agents(1).l);
realDataSet=Agents(1).realdataset;
if usejava('desktop')
    wbVisibility=true;
else
    wbVisibility=false;
end

top_dir_path=strcat('results/HO');
folder_name=strcat(num2str(M),'_a_',num2str(Agents(1).TotalNumLevel),'_pl');
full_path=strcat(top_dir_path,'/',folder_name,'/');
inputDim=size(Agents(1).X,1);
thetas=zeros(inputDim+2,M);
pxADMM_fd_flag=1;
iterCount=0;
% the outer iteration begains here
updated_z=[Agents(1).sigma_f;Agents(1).l;Agents(1).sigma_n];
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
    IterCounts{m}=1;
end
Steps=cell(M,1);
for m=1:M
    Steps{m}=[];
end
outSteps=[];
sub_old_z=zeros(inputDim+2,M);
for m=1:M
    sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
end
if sync==1
    %% sync
    if wbVisibility
        wb=waitbar(0,'Preparing','Name','pxADMM_{fd,sync}');
        set(wb,'color','w');
    end
    while pxADMM_fd_flag>0
        iterCount=iterCount+1;

        if mod(iterCount,250)==0
            disp(iterCount);
            disp(step);
        end

        Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
        old_z=updated_z;
        updated_z=zeros(inputDim+2,1);
        global_step=0;


        parfor m=1:M
            Agents(m)=Agents(m).runPxADMM_fd;
            Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]];
            step_m=max(vecnorm([Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]-sub_old_z(:,m),2,2));
            %             step_m=max(vecnorm(Agents(m).l-sub_old_z(2:end,m),2,2));
            SubSteps{m}=[SubSteps{m},step_m];
            global_step=max(global_step,step_m);
        end
        for m=1:M
            Agents(m).action_status=1;
            updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];

            sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
            %     Agents(m).rho=Agents(m).rho*0.9999;
        end
        updated_z=updated_z/M;
        step = max(vecnorm(global_step,2,2));
%         step = max(vecnorm(updated_z(1:end)-old_z(1:end),2,2));
        outSteps=[outSteps,step];

        if wbVisibility
            waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM_{fd}: ',iterCount/maxIter*100,'% , step:', step))
        end
        if step < epsilon
            pxADMM_fd_flag=0;
        end
        if iterCount>=maxIter
            pxADMM_fd_flag=0;
        end
    end
    sigma_pxADMM_fd=updated_z(1);
    l_pxADMM_fd=updated_z(2:end-1);
    sigma_n_pxADMM_fd=updated_z(end);
    for temp_for=1
        gcf=figure;
        cvgValue=Inf*ones(1,inputDim+2);

        tiledlayout(D+2,1,'TileSpacing','Compact','Padding','Compact');
        for i=1:inputDim+2
            %         subplot(inputDim+2,1,i)
            nexttile(i);
            if i==1
                ylabel('\sigma_f')
            elseif i==D+2
                ylabel('\sigma_n')
            else
                txt=strcat('l_',num2str(i-1));
                ylabel(txt)
            end
            hold on
            for m=1:M

                cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
            end
            y_c=yline(cvgValue(i),'-.b');
            if realDataSet==0
                y_r=yline(Agents(1).realz(i),'r-.');
            end
            for m=1:M
                plot(Zs{m}(i,:));
            end

            set(gca, 'XScale', 'log')
            hold off

            if i==1

                if realDataSet==0
                    legendTxt=cell(2,1);
                    legendTxt{1}='converged value';
                    legendTxt{2}='real hyperparameter value';
                    lgd=legend([y_c;y_r],legendTxt,'Location','northoutside','Orientation', 'Horizontal');
                else
                    lgd=legend(y_c,'converged value','Location','northoutside','Orientation', 'Horizontal');
                end
                sgtitle('pxADMM_{fd,sync}^* - hyperparameters')
            end
            %     set(gca,'XScale','log')
        end
        s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=600;
s.Format='png';
s.Width=5;
s.Height=5;
s.FontSizeMin=14;
s.Format='png';
        fname=strcat(full_path,'pxADMM_fd_sync_vars');
        fname=strcat(fname,'_',num2str(M),'_agents');
        hgexport(gcf,fname,s);

        close gcf;

        outSteps=SubSteps;

        gcf=figure;
        tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
        nexttile(1)
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
        lgd=legend(legendText,'Location','northoutside','Orientation', 'Horizontal');
        %     lgd.Layout.Tile = 'north';
        lgd.NumColumns=4;

        xlabel('steps')
        ylabel('step size')
        sgtitle('pxADMM_{fd,sync}^* - step size')
        set(gca,'YScale','log')
        set(gca,'XScale','log')
        hold off;
        s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=600;
s.Format='png';
s.Width=5;
s.Height=5;
s.FontSizeMin=14;
s.Format='png';
        fname=strcat(full_path,'pxADMM_fd_sync_Steps');
        fname=strcat(fname,'_',num2str(M),'_agents');
        hgexport(gcf,fname,s);
        close gcf;
    end

    
elseif sync==0
    %% async
    stepForStops=[];
    if wbVisibility
        wb=waitbar(0,'Preparing','Name','pxADMM_{fd,async}');
        set(wb,'color','w');
    end
    P01s=rand(1,M)*0.6+0.4;
    AgentsActionStatus=[];
    while pxADMM_fd_flag
        % Randomly activate some of the agents
        activatedAgents=zeros(M,1);
        activated_agents=[];
        for m=1:M
            if Agents(m).action_status||Agents(m).communicationAbility
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
        iterCount=iterCount+1;
        if ~mod(iterCount,250)
            disp(iterCount);
            disp(step);
        end
        Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
        old_z=updated_z;
        updated_z=zeros(inputDim+2,1);
        global_step=0;


        for i=1:length(activated_agents)
            m=activated_agents(i);
            Agents(m)=Agents(m).runPxADMM_fd;
            Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]];
            step_m=max(vecnorm([Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]-sub_old_z(:,m),2,2));
            %             step_m=max(vecnorm(Agents(m).l-sub_old_z(2:end,m),2,2));
            SubSteps{m}=[SubSteps{m},step_m];
            global_step=max(global_step,step_m);
        end
        for m=1:M
            updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];

            sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
            %     Agents(m).rho=Agents(m).rho*0.9999;
        end
        updated_z=updated_z/M;
%         step = max(vecnorm(global_step,2,2));
        step = max(vecnorm(global_step,2,2));
        %         step = max(vecnorm(updated_z-old_z,2,2));
        outSteps=[outSteps,step];
        if iterCount>5
            stepForStop=mean(outSteps(end-5:end));
        else
            stepForStop=mean(outSteps(1:end));
        end
        stepForStops=[stepForStops;stepForStop];

        if wbVisibility
            waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM_{fd,async}: ',iterCount/maxIter*100,'% , step:', step))
        end
        if stepForStop < epsilon
            pxADMM_fd_flag=0;
        end
        if iterCount>=maxIter
            pxADMM_fd_flag=0;
        end
    end
    outSteps=stepForStops;
    sigma_pxADMM_fd=updated_z(1);
    l_pxADMM_fd=updated_z(2:(2+inputDim-1));
    sigma_n_pxADMM_fd=updated_z(end);
    for temp_for=1
        % Plot
        gcf=figure;
        cvgValue=Inf*ones(1,inputDim+2);

        tiledlayout(D+2,1,'TileSpacing','Compact','Padding','Compact');
        for i=1:inputDim+2
            %         subplot(inputDim+2,1,i)
            nexttile(i);
            if i==1
                ylabel('\sigma_f')
            elseif i==D+2
                ylabel('\sigma_n')
            else
                txt=strcat('l_',num2str(i-1));
                ylabel(txt)
            end
            hold on
            for m=1:M

                cvgValue(i)=min(cvgValue(i),Zs{m}(i,end));
            end
            y_c=yline(cvgValue(i),'-.b');
            if realDataSet==0
                y_r=yline(Agents(1).realz(i),'r-.');
            end
            for m=1:M
                plot(Zs{m}(i,:));
            end

            set(gca, 'XScale', 'log')
            hold off

            if i==1

                if realDataSet==0
                    legendTxt=cell(2,1);
                    legendTxt{1}='converged value';
                    legendTxt{2}='real hyperparameter value';
                    lgd=legend([y_c;y_r],legendTxt,'Location','northoutside','Orientation', 'Horizontal');
                else
                    lgd=legend(y_c,'converged value','Location','northoutside','Orientation', 'Horizontal');
                end
                sgtitle('pxADMM_{fd,async}^* - hyperparameters')
            end
        end
        s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=600;
s.Format='png';
s.Width=5;
s.Height=5;
s.FontSizeMin=14;
s.Format='png';
        fname=strcat(full_path,'pxADMM_fd_async_vars');
            fname=strcat(fname,'_',num2str(M),'_agents');
        hgexport(gcf,fname,s);
        close gcf;

        gcf=figure;
        for m=1:M
            subplot(M,1,m);
            status=kron(AgentsActionStatus(m,1:50),ones(1,10));
            area(linspace(0,51,length(status)),status);
        end

        outSteps=SubSteps;
        close gcf

        gcf=figure;
        tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
        nexttile(1)
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
        lgd=legend(legendText,'Location','northoutside','Orientation', 'Horizontal');
        %     lgd.Layout.Tile = 'north';
        lgd.NumColumns=4;


        xlabel('steps')
        ylabel('step size')
        sgtitle('pxADMM_{fd,async}^* - step size')
        set(gca,'YScale','log')
        set(gca,'XScale','log')
        hold off;

        fname=strcat(full_path,'pxADMM_fd_async_Steps');
            fname=strcat(fname,'_',num2str(M),'_agents');
        hgexport(gcf,fname,s);
        %     saveas(gcf,fname,'png');
        close gcf;
    end

end
for m=1:M
    thetas(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
end
if wbVisibility
    delete(wb);
end
end

