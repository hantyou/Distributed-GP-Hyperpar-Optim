function field1=gen2D_GP(m,n,l,sigma_f,range_x,range_y)
%GEN2D_GP generate 2D GP field, the function is based on code credit below:
% Copyright (c) 2016, Zdravko Botev
% All rights reserved.
% Code can be found on https://web.maths.unsw.edu.au/~zdravkobotev/
%The function is explained below:
% m, n: respectively the vertical and horizontal data points number
% l: characteristic scale, cannot be larger than the field range
% sigma_f: kernel variance 
% range_x, range_y: respectively the field range of horizon and vertice
plotFlag=0;

tx=linspace(range_x(1),range_x(2),m)'; 
ty=linspace(range_y(1),range_y(2),n)'; % create grid for field
Rows=zeros(m,n); Cols=Rows;
for j=1:m % sample covariance function at grid points;
    for i=1:n
        Rows(j,i)=k2(tx(j)-tx(1),ty(i)-ty(1),sigma_f,l); % rows of blocks of cov matrix
        Cols(j,i)=k2(tx(1)-tx(j),ty(i)-ty(1),sigma_f,l); % columns of blocks of cov matrix
    end
end
% create the first row of the block circulant matrix with circular blocks
% and store it as a matrix suitable for fft2;
BlkCirc_row=[Rows, Cols(:,end:-1:2);
    Cols(end:-1:2,:), Rows(end:-1:2,end:-1:2)];
% compute eigen-values
lam=real(fft2(BlkCirc_row))/(2*m-1)/(2*n-1);
if abs(min(lam(lam(:)<0)))>10^-15
%     error('Could not find positive definite embedding!')
else
    lam(lam(:)<0)=0; lam=sqrt(lam);
end
% generate field with covariance given by block circular matrix
F=fft2(lam.*complex(randn(2*m-1,2*n-1),randn(2*m-1,2*n-1)));
F=F(1:m,1:n); % extract subblock with desired covariance
field1=real(F); 
% field2=imag(F); % two independent fields with desired covariance

if plotFlag
    [mesh_x,mesh_t]=meshgrid(tx,ty);
    surf(mesh_x',mesh_t',field1,'edgecolor','none')
    xlabel('x');
    ylabel('y');
    zlabel('z')
end
end

function a=k2(x,t,sigma_f,l)
    l=diag(l);
    a=sigma_f^2*exp(-norm([inv(l)*x,inv(l)*t])^2/2);
end
