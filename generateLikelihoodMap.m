function LL=generateLikelihoodMap(X,Z,theta_range,sigma_n)
%GENERATELIKELIHOODMAP Summary of this function goes here
%   Detailed explanation goes here

[H_Num,~]=size(theta_range);

[Dim,D_Num]=size(X);

sigma_fs=linspace(theta_range(1,1),theta_range(1,2),50);
l1s=linspace(theta_range(2,1),theta_range(2,2),50);
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

% NLL=log(-LL)/log(10);
NLL=-LL;
figure,imshow(NLL,[]);
gcf=figure;
contour(l1s,sigma_fs,NLL,'--k','LineWidth',2);
hold on;
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
% set(gca, 'ZScale', 'log')
imagesc(l1s,sigma_fs,NLL);
contour(l1s,sigma_fs,NLL,'--k','LineWidth',2);
hold off;
fname='results/LikelihoodHyperpars';
saveas(gcf,fname,'png');
close gcf;
end