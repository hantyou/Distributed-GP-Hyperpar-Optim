function [X_m,y_m,Fv_iterations]= FindInducingPoints(X,y,m_ind,theta,sigma_n)
%FINDIDUCINGPOINTS Summary of this function goes here
%   X is the full training input set
%   y is the full training output set
%   m_ind is the number of inducing points
%   sigma_n is the noise std

%% Extract information from input
[D,N]=size(X);
[output_D,~]=size(y);
sigma_f=theta(1);
l=theta(2:(2+D-1));
%% Initialization and data prepare
X_m=[];
X_n_m=X;
y_m=[];
y_n_m=y;
point_left_to_add=m_ind;
K_NN=getK(X,X,theta,sigma_n);
inv_K_NN=inv(K_NN);
idx_m=[];
idx_n_m=1:N;
idx_test=[];
Fv_iterations=[];

%% iteration begins
while point_left_to_add>0
    [~,n_m]=size(X_n_m);
    Fv_values=zeros(1,n_m);
    parfor i=1:n_m
        x_test=X_n_m(:,i);
        y_test=y_n_m(:,i);
        X_m_test=[X_m,x_test];
        y_m_test=[y_m,y_test];
        idx_m_test=[idx_m,idx_n_m(i)];
        K_MM_test=K_NN(idx_m_test,idx_m_test);
        K_NM_test=K_NN(:,idx_m_test);
        Q_NN_test=K_NM_test/K_MM_test*K_NM_test';
        K_tilde_test=K_NN-Q_NN_test;
        Fv_values(i)=F_v(y,Q_NN_test,K_tilde_test,sigma_n);
    end
    [max_Fv,id]=max(Fv_values);
    X_m=[X_m,X_n_m(:,id)];
    y_m=[y_m,y_n_m(:,id)];
    Fv_iterations=[Fv_iterations,max_Fv];
    X_n_m(:,id)=[];
    y_n_m(:,id)=[];
    point_left_to_add=point_left_to_add-1;
end



end

