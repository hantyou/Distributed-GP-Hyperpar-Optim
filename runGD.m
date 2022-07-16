function [newSigma,newL,Steps] = runGD(Agents,M,initial_sigma_f,initial_l,sigma_n,stepSize,stopCriteria,maxIter)
%RUNGD Summary of this function goes here
%   Detailed explanation goes here

if usejava('desktop')
    wbVisibility=true;
else
    wbVisibility=false;
end
D=length(initial_l);
sampleSize=M*length(Agents(1).Z);
epsilon=stopCriteria;
GDflag=1;
Sigmas=zeros(maxIter,1);
Sigmas(1)=initial_sigma_f;
Sigma_ns=zeros(maxIter,1);
Sigma_ns(1)=sigma_n;
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
    Agents(m) = Agents(m).initLocalGD(initial_sigma_f,initial_l',sigma_n,stepSize);
    Agents(m).sigma_n=sigma_n;
    Agents(m).pd_l=zeros(D,1);
end
% perform GD
if wbVisibility
    wb=waitbar(0,'Preparing','Name','GD');
    set(wb,'color','w');
end
while GDflag
    iterCount=iterCount+1;
    oldSigma=Sigmas(iterCount-1);
    oldL=Ls(:,iterCount-1);
    oldSigma_n=Sigma_ns(iterCount-1);
    newSigma=Sigmas(iterCount-1);
    newL=Ls(:,iterCount-1);
    newSigma_n=Sigma_ns(iterCount-1);
    %     newSigma_n=Sigma_ns(iterCount-1);
    newLikelihood=0;
    %     spmd(M)
    %         obj=Agents(labindex);
    %         obj = obj.runLocalGD;
    %     end
    %     Agents=[obj{:}];
    for m=1:M
        Agents(m) = Agents(m).runLocalGD;
        newSigma=newSigma-Agents(m).mu*Agents(m).pd_sigma_f;
        newL=newL-Agents(m).mu*Agents(m).pd_l;
        newSigma_n=newSigma_n-Agents(m).mu*Agents(m).pd_sigma_n;
        newLikelihood=newLikelihood+Agents(m).Z'*inv(Agents(m).K)*Agents(m).Z+log(det(Agents(m).K));
    end

    Sigmas(iterCount)=newSigma;
    Ls(:,iterCount)=newL;
    Sigma_ns(iterCount)=newSigma_n;
    Likelihoods(iterCount)=-0.5*newLikelihood+0.5*sampleSize*log(2*pi);
    for m=1:M
        Agents(m).sigma_f=newSigma;
        Agents(m).l=newL;
        Agents(m).sigma_n=newSigma_n;
        Agents(m).z=[newSigma;newL;newSigma_n];
        Zs{m}=[Zs{m},Agents(m).z];
    end
    %         step=max(abs(Sigmas(iterCount)-Sigmas(iterCount-1)),abs(Ls(1,iterCount)-Ls(1,iterCount-1)));
    step=max(vecnorm([newSigma;newL;newSigma_n]-[oldSigma;oldL;oldSigma_n],2,2))  ;
    Steps(iterCount)=step;

    if wbVisibility
        waitbar(iterCount/maxIter,wb,sprintf('%s %.2f %s %f','GD: ',iterCount/maxIter*100,'% , step:', step))
    end

    if step<epsilon
        GDflag=0;
    end
    if iterCount>=maxIter
        GDflag=0;
    end
end

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
% nexttile(1);
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

    sgtitle('GD - hyperparameters')
    hold off
end




s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname='results/GD_vars';
fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
close gcf

gcf=figure;
tiledlayout(1,1,'TileSpacing','Compact','Padding','Compact');
nexttile(1);
semilogy(Steps)
set(gca,'XScale','log')
xlabel('steps')
ylabel('step size')
title('GD - step size')
s=hgexport('factorystyle');
s.Resolution=600;
s.Format='png';
fname='results/GD_Steps';
fname=strcat(fname,'_',num2str(M),'_agents');
hgexport(gcf,fname,s);
close gcf
if wbVisibility
    delete(wb);
end
end

