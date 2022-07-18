function [sigma_ADMM_fd,l_ADMM_fd,sigma_n_ADMM_fd,Steps,IterCounts] = runADMM_fd(Agents,M,stepSize,epsilon,maxOutIter,maxInIter)
%RUNADMM Summary of this function goes here
%   Detailed explanation goes here
if usejava('desktop')
    wbVisibility=true;
else
    wbVisibility=false;
end
D=length(Agents(1).l);
outADMMflag=1;
outIterCount=0;
% the outer iteration begains here
updated_z=[Agents(1).sigma_f;Agents(1).l;Agents(1).sigma_n];
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
if wbVisibility
    wb=waitbar(0,'Preparing','Name','ADMM_{fd}');
    set(wb,'color','w');
end
sub_old_z=zeros(D+2,M);
for m=1:M
    sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
end
while outADMMflag
    outIterCount=outIterCount+1;
    Agents=Agents.obtain_data_from_Neighbor_ADMM_fd;
    global_step=0;
    updated_z=0;
    parfor m=1:M
        %         [Agents(m),~,~,~]=Agents(m).runLocalInnerADMM_fd(maxInIter,epsilon);
        [Agents(m),Zs_m,Steps_m,inIterCount_m]=Agents(m).runLocalInnerADMM_fd(maxInIter,epsilon);
        Zs{m}=[Zs{m},Zs_m];
        Steps{m}=[Steps{m},Steps_m];
        IterCounts{m}=[IterCounts{m},IterCounts{m}(outIterCount)+inIterCount_m];
        step_m=norm([Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]-sub_old_z(:,m));
        global_step=max(global_step,step_m);
        sub_old_z(:,m)=[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
        updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n];
    end
    updated_z=updated_z/M;
    step = global_step;
    outSteps=[outSteps,step];

if wbVisibility
    waitbar(outIterCount/maxOutIter,wb,sprintf('%s %.2f %s %f','ADMM_{fd}: ',outIterCount/maxOutIter*100,'% , step:', step))
    end
    if step < epsilon
        outADMMflag=0;
    end
    if outIterCount>=maxOutIter
        outADMMflag=0;
    end
end
sigma_ADMM_fd=updated_z(1);
l_ADMM_fd=updated_z(2:(2+D-1));
sigma_n_ADMM_fd=updated_z(end);
% Plot
gcf=figure;
tiledlayout(D+2,1,'TileSpacing','Compact','Padding','Compact');
realDataSet=Agents(1).realdataset;
for z_i=1:(D+2)
    nexttile(z_i);
    hold on
    y_c=yline(Zs{1}(z_i,end),'b-.');
    if realDataSet==0
        y_r=yline(Agents(1).realz(z_i),'r-.');
    end
    for m=1:M
        plot(Zs{m}(z_i,:));
    end
    xlabel('steps')
    set(gca,'XScale','log')
    if z_i==1
        ylabel('\sigma_f');
    elseif z_i==D+2
        ylabel('\sigma_n');
    else
        ylabel(strcat('l_',num2str(z_i-1)));
    end
    if z_i==1

        if realDataSet==0
            legendTxt=cell(2,1);
            legendTxt{1}='converged value';
            legendTxt{2}='real hyperparameter value';
            lgd=legend([y_c;y_r],legendTxt,'Location','northoutside','Orientation', 'Horizontal');
        else
            lgd=legend(y_c,'converged value','Location','northoutside','Orientation', 'Horizontal');
        end
%         lgd.Layout.Tile = 'north';
    end

    sgtitle('ADMM_{fd} - hyperparameters')
    hold off
end
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
top_dir_path=strcat('results/HO');
folder_name=strcat(num2str(M),'_a_',num2str(Agents(1).TotalNumLevel),'_pl');
full_path=strcat(top_dir_path,'/',folder_name,'/');
fname=strcat(full_path,'ADMM_fd_vars');
% fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
close gcf
%
gcf=figure;
%% below there is bug
semilogy(IterCounts{1}(2:end),outSteps);
set(gca,'XScale','log')
xlabel('steps')
ylabel('step size')
title('ADMM_{fd} - step size')
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname=strcat(full_path,'ADMM_fd_Steps');
% fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
close gcf
Steps=outSteps;
if wbVisibility
    delete(wb);
end

end

