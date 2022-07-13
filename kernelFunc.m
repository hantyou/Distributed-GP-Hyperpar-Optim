function k=kernelFunc(x1,x2,theta,sigma_n,type)
switch type
    case {'RBF','rbf'}
        D=length(x1);
        sigma_f=theta(1);
        l=theta(2:(2+D-1));
        k=sigma_f^2*exp(-0.5*(x1-x2)'*inv(diag(l.^2))*(x1-x2));
        if x1==x2
            k=k+sigma_n^2;
        end
end


end