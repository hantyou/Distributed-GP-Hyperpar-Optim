function [Mean,Uncertainty] = GPR_predict(X,Z,theta,range,sigma_n,plotFlag)
%GPR_PREDICT Summary of this function goes here
%   Detailed explanation goes here
range_x1=range(1,:);
range_x2=range(2,:);
sigma_f=theta(1);
l=theta(2:end);
N=size(X,2);

% X=subDataSetsX(:,:,1);

% Z=subDataSetsZ(1,:);
% N=sampleSize/M;

K=zeros(N,N);
% sigma_f=sigma_pxADMM_fd_sync;
% l=l_pxADMM_fd_sync;
% l=newL;
for i=1:N
    for j=1:N
        K(i,j)=k(X(:,i),X(:,j),sigma_f,l,sigma_n);
    end
end

y=Z';

mean=[];
var=[];
ts_1=linspace(range_x1(1),range_x1(2),100);
ts_2=linspace(range_x2(1),range_x2(2),100);
try
L=chol(K)';
[mesh_x,mesh_y]=meshgrid(ts_1,ts_2);
[h,w]=size(mesh_x);
mean=zeros(h,w);
var=zeros(h,w);
alpha=L'\(L\y);
parfor xi=1:h
    for yi=1:w
        x_star=[mesh_x(xi,yi);mesh_y(xi,yi)];
        [y_est,y_var_est] = GPRSinglePointPredict(X,x_star,alpha,L,theta,sigma_n);
        
        mean(xi,yi)=y_est;
        var(xi,yi)=y_var_est;
    end
end
catch
invK=inv(K);
[mesh_x,mesh_y]=meshgrid(ts_1,ts_2);
[h,w]=size(mesh_x);
mean=zeros(h,w);
var=zeros(h,w);
for xi=1:h
    for yi=1:w
        x_star=[mesh_x(xi,yi);mesh_y(xi,yi)];
        
        K_star_star=k(x_star,x_star,sigma_f,l,sigma_n);
        K_star=zeros(1,N);
        for i=1:N
            K_star(i)=k(x_star,X(:,i),sigma_f,l,sigma_n);
        end
        y_est=K_star*invK*y;
        y_var_est=K_star_star-K_star*invK*K_star';

        mean(xi,yi)=y_est;
        var(xi,yi)=y_var_est;
    end
end
end

if plotFlag==1
    figure,subplot(121)
    surf(mesh_x,mesh_y,(mean),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    scatter3(X(1,:),X(2,:),y,'r*')
    hold off;
    xlabel('x1')
    ylabel('x2')
    zlabel('y')
    title('GPR result - mean')
    
    subplot(122)
    surf(mesh_x,mesh_y,(var),'edgecolor','none','FaceAlpha',0.9);
    hold on,
    ax = gca;
    ax.YDir = 'normal';
    Z_ps=interp2(mesh_x,mesh_y,(var),X(1,:),X(2,:));
    scatter3(X(1,:),X(2,:),Z_ps,'r*')
    set(gca,'ZScale','log')
    hold off;
    xlabel('x1')
    ylabel('x2')
    zlabel('y')
    title('GPR result - variance (in log plot)')
end
% end
Mean=mean;
Uncertainty=var;
end

