clc,clear;
close all;
% x=[-1.5 -1 -0.75 -0.4 -0.25 0];
% y=[-1.7 -1.1 -0.4 0.1 0.4 0.9]';


x=unifrnd(-10,10,2,500);
sigma_n=0.05;
y=sin(2*vecnorm(x))'.*exp(-vecnorm(x)'/6);
% y=exp(-vecnorm(x).^2./4)';
sigma_f=1.27;
% sigma_f=1;
% l=sqrt(-0.125/log(1.42/(1.7-sigma_n^2)));
l=1;

N=length(x);
K=zeros(N,N);
for i=1:N
    for j=1:N
        K(i,j)=k(x(:,i),x(:,j),sigma_f,l,sigma_n);
    end
end

x_star=-0.2;
K_star_star=k(x_star,x_star,sigma_f,l,sigma_n);
K_star=zeros(1,N);
for i=1:N
    K_star(i)=k(x_star,x(:,i),sigma_f,l,sigma_n);
end

y_est=K_star*inv(K)*y;
y_var_est=K_star_star-K_star*inv(K)*K_star';

mean=[];
var=[];
ts=-10:0.033:10;
invK=inv(K);
L=chol(K)';
[mesh_x,mesh_y]=meshgrid(ts,ts);
[h,w]=size(mesh_x);
mean=zeros(h,w);
var=zeros(h,w);
parfor xi=1:h
    for yi=1:w
        x_star=[mesh_x(xi,yi);mesh_y(xi,yi)];
        K_star_star=k(x_star,x_star,sigma_f,l,sigma_n);
        K_star=zeros(1,N);
        for i=1:N
            K_star(i)=k(x_star,x(:,i),sigma_f,l,sigma_n);
        end
        alpha=L'\(L\y);
        y_est=K_star*alpha;
    %     y_est=K_star*inv(K)*y;
        v=L\K_star';
        y_var_est=K_star_star-v'*v;
    %     y_var_est=K_star_star-K_star*invK*K_star';
        mean(xi,yi)=y_est;
        var(xi,yi)=y_var_est;
    end
end
%%
surf(mesh_x,mesh_y,(mean).*(2^8),'edgecolor','none','FaceAlpha',0.9);
hold on,
ax = gca;
ax.YDir = 'normal';
scatter3(x(1,:),x(2,:),y.*(2^8),'r.')
hold off;

figure,
surf(mesh_x,mesh_y,(var).*(2^8),'edgecolor','none','FaceAlpha',0.9);
hold on,
ax = gca;
ax.YDir = 'normal';
Z_ps=interp2(mesh_x,mesh_y,(var).*(2^8),x(1,:),x(2,:));
scatter3(x(1,:),x(2,:),Z_ps,'r.')
set(gca,'ZScale','log')
hold off;


function a=k(x1,x2,sigma_f,l,sigma_n)
if x1~=x2
    a=sigma_f^2*exp(-norm(x1-x2)^2/2/l^2);
else
    a=sigma_f^2*exp(-norm(x1-x2)^2/2/l^2)+sigma_n^2;
end
end

% % 
% function a=k(x1,x2,sigma_f1,l1,sigma_n)
% sigma_f2=0.5*sigma_f1;
% l2=0.1*l1;
% if x1~=x2
%     a=sigma_f1^2*exp(-norm(x1-x2)^2/2/l1^2)+sigma_f2^2*exp(-norm(x1-x2)^2/2/l2^2);
% else
%     a=sigma_f1^2*exp(-norm(x1-x2)^2/2/l1^2)+sigma_f2^2*exp(-norm(x1-x2)^2/2/l2^2)+sigma_n^2;
% end
% end


% function a=k(x1,x2,sigma_f1,l1,sigma_n)
% sigma_f2=0.2*sigma_f1;
% v=0.13;
% l2=0.05*l1;
% if x1~=x2
%     a=sigma_f1^2*exp(-(x1-x2)^2/2/l1^2)+exp(-2*sin(v*pi*(x1-x2))^2);
% else
%     a=sigma_f1^2*exp(-(x1-x2)^2/2/l1^2)+exp(-2*sin(v*pi*(x1-x2))^2)+sigma_n^2;
% end
% end
