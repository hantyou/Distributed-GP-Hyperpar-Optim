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
if usejava('desktop')
    wbVisibility=true;
else
    wbVisibility=false;
end

if wbVisibility
    wb=waitbar(0,'Preparing','Name','ADMM');

    set(wb,'color','w');
end
rho_0=Agents(1).rho;
maxInIter_0=maxInIter;
while outADMMflag
    outIterCount=outIterCount+1;
    %     disp(outIterCount)
    % Calculate z from agents' data
    old_z=updated_z;
    updated_z=zeros(size(updated_z));
    for m=1:M
        %         Agents(m).mu=stepSize;
        updated_z=updated_z+[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]+1/Agents(m).rho*Agents(m).beta;
    end
    updated_z=1/M*updated_z;
    % transmit z from center to agents
    for m=1:M
        Agents(m).z=updated_z;
    end

    step = norm(updated_z-old_z);
    outSteps=[outSteps,step];
    % at each agent, perform inner iteration
    % for each inner iteration, the theta=[sigma_f, l] is going to be
    % updated, and beta is also to be updated
    parfor m=1:M
        [Agents(m),Zs_m,Steps_m,inIterCount_m]=Agents(m).runLocalInnerADMM(maxInIter,epsilon);
        Zs{m}=[Zs{m},Zs_m];
        Steps{m}=[Steps{m},Steps_m];
        IterCounts{m}=[IterCounts{m},IterCounts{m}(outIterCount)+inIterCount_m];
    end

    if wbVisibility
        waitbar(outIterCount/maxOutIter,wb,sprintf('%s %.2f %s %f','ADMM: ',outIterCount/maxOutIter*100,'% , step:', step))
    end
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
gcf=figure;
tiledlayout(D+2,1,'TileSpacing','Compact','Padding','Compact');
realDataSet=Agents(1).realdataset;
for z_i=1:(D+2)
    nexttile(z_i);
    hold on
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
    y_c=yline(Zs{1}(z_i,end),'b-.');
    if realDataSet==0
        y_r=yline(Agents(1).realz(z_i),'r-.');
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

    sgtitle('ADMM - hyperparameters')
    hold off
end
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname='results/ADMM_vars';
fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
%
gcf=figure;
%% below there is bug
semilogy(IterCounts{1}(2:end),outSteps);
set(gca,'XScale','log')
xlabel('steps')
ylabel('step size')
title('ADMM - step size')
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname='results/ADMM_Steps';
fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);


Steps=outSteps;

if wbVisibility
    delete(wb);
end

end

