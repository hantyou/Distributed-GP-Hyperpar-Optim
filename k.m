function a=k(x1,x2,sigma_f,l,sigma_n)
inputDim=length(x1);
Sigma=diag(l).^2;
invSigma=Sigma\eye(inputDim);
if norm(x1-x2)>0
    a=sigma_f^2*exp(-0.5*(x1-x2)'*invSigma*(x1-x2));
else
    a=sigma_f^2*exp(-0.5*(x1-x2)'*invSigma*(x1-x2))+sigma_n^2;
end
end