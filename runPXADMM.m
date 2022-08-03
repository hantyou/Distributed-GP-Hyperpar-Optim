function [sigma_pxADMM,l_pxADMM,sigma_n_pxADMM,Steps,Zs] = runPXADMM(Agents,M,epsilon,maxIter)
%RUNPXADMM Summary of this function goes here
%   Detailed explanation goes here
if usejava('desktop')
    wbVisibility=true;
else
    wbVisibility=false;
end
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
if wbVisibility
wb=waitbar(0,'Preparing','Name','pxADMM');
set(wb,'color','w');
end
while pxADMMflag
    iterCount=iterCount+1;
    if ~mod(iterCount,500)
    disp(iterCount);
    disp(step);
    end
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
    parfor m=1:M
        Agents(m)=Agents(m).runLocalPxADMM;
        Zs{m}=[Zs{m},[Agents(m).sigma_f;Agents(m).l;Agents(m).sigma_n]];
    end
    step = norm(updated_z(1:end)-old_z(1:end));
    Steps=[Steps,step];
    
if wbVisibility
   waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','pxADMM: ', iterCount/maxIter*100,'% , step:', step))
end
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

    sgtitle('pxADMM - hyperparameters')
    hold off
end
s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=600;
s.Format='png';
s.Width=5;
s.Height=5;
s.FontSizeMin=14;
s.Format='png';
top_dir_path=strcat('results/HO');
folder_name=strcat(num2str(M),'_a_',num2str(Agents(1).TotalNumLevel),'_pl');
full_path=strcat(top_dir_path,'/',folder_name,'/');
fname=strcat(full_path,'pxADMM_vars');
% fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
close gcf

gcf=figure;
tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
nexttile(1);
semilogy(Steps)
set(gca,'XScale','log')
xlabel('steps')
ylabel('step size')
title('pxADMM convergence')
fname=strcat(full_path,'pxADMM_Steps');
% fname=strcat(fname,'_',num2str(M),'_agents');
s=hgexport('factorystyle');
s.LineWidthMin=1.2;
s.Resolution=600;
s.Format='png';
s.Width=5;
s.Height=5;
s.FontSizeMin=14;
s.Format='png';
hgexport(gcf,fname,s)
close gcf;


if wbVisibility
delete(wb);
end

end

