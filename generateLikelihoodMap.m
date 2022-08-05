function LL=generateLikelihoodMap(X,Z,theta_range,sigma_n)
%GENERATELIKELIHOODMAP Summary of this function goes here
%   Detailed explanation goes here

[H_Num,~]=size(theta_range);

[Dim,D_Num]=size(X);

sigma_fs=linspace(theta_range(1,1),theta_range(1,2),40);
l1s=linspace(theta_range(2,1),theta_range(2,2),40);
sigma_fs=10.^sigma_fs;
l1s=10.^l1s;

sf_Num=length(sigma_fs);
l_Num=length(l1s);

LL=zeros(sf_Num,l_Num);

for m=1:sf_Num
    m
    parfor n=1:l_Num
        sigma_f=sigma_fs(sf_Num-m+1);
        l1=l1s(n);
        theta=[sigma_f;l1;l1];
        K=getK(X,theta,sigma_n);
        LL(m,n)=-1/2*Z/K*Z'-1/2*log(det(K))-D_Num/2*log(2*pi);
    end
end

NLL=log(-LL)/log(20);
[min_x,min_y]=find(NLL==min(NLL(:)));
min_sigma_f=sigma_fs(sf_Num-min_x+1);
min_l=l1s(min_y);
% NLL=-LL;
% gcf=figure;
% imshow(NLL,[]);
% close gcf;

% gcf=figure;
% contour(l1s,flip(sigma_fs),NLL,10,'--k','LineWidth',2,'ShowText','on');
% hold on;
% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')
% % set(gca, 'ZScale', 'log')
% imagesc(l1s,flip(sigma_fs),NLL);
% colorbar
% contour(l1s,flip(sigma_fs),NLL,10,'--k','LineWidth',2,'ShowText','on');
% hold off;
% fname='results/LikelihoodHyperpars';
% saveas(gcf,fname,'png');
% close gcf;

gcf=figure;
% ax1=axes;
% imagesc(ax1,l1s,flip(sigma_fs),NLL);
% surf(ax1,l1s,flip(sigma_fs),NLL,'edgecolor','none','facecolor','interp');
% colormap(ax1,'gray');
% hold on;
ax2=axes;
contour(ax2,l1s,flip(sigma_fs),NLL,50);
hold on ;
scatter(min_l,min_sigma_f,50,'^k','filled')
text(min_l-0.11,min_sigma_f-0.15,strcat("(",num2str(min_l,3),",",num2str(min_sigma_f,3),")"),"EdgeColor","k","Color","k",'HorizontalAlignment','right','BackgroundColor','w');
colormap(ax2,'jet');
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
xlabel('$l$','Interpreter','latex','FontSize',17);
ylabel('$\sigma_f$','Interpreter','latex','FontSize',17);
colorbar;
fname='results/LikelihoodHyperparsOnlyContour';
saveas(gcf,fname,'png');
close gcf;
% linkaxes([ax1,ax2])

% ax2.Visible = 'off';

end