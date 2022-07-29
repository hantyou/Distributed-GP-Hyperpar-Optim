classdef agent

    properties
        % Info about agent itself
        Code % agent serial number m, {1,2,...,M}
        Position % Current position of agent
        M
        action_status
        commuRange
        % Topology information
        Neighbors
        N_size
        A
        % async behavior
        updatedVars
        updatedVarsNumber
        updatedVarsNumberThreshold
        communicationAbility
        slowSync
        slowSyncThreshold

        % Kernel parameters and data
        sigma_f % Kernel variance
        l % Kernel characteristic length
        idx % idx of local data set in global data set
        X % Local training data set input
        Z % Local training data set target
        distX1
        distX2
        distXd
        sigma_n % Noise variance
        X_induced
        Z_induced

        % GD parameters
        mu % Step size for GD

        % ADMM parameters
        beta
        z
        rho % 500 in the papaer

        % pxADMM parameters
        L

        % ADMM_fd parameters
        z_mn
        beta_mn
        z_nm
        beta_nm

        theta_n
        beta_n


        % Common parameters, matrix, etc.
        K % The local kernel matrix
        N_m % The local data set size for agent m
        pd_sigma_f % partial derivative of sigma_f
        pd_l % partial derivative of l
        pd_sigma_n % partial derivative of sigma_n
        % Other
        Transmission
        Zs
        Steps
        currentStep
        maxIter
        runningHO
        realz
        realdataset
        TotalNumLevel
    end

    methods
        function obj = agent(Code)
            if nargin>0
                obj.Code=Code;
            end
        end

        function obj = initLocalGD(obj,sigma_f,l,sigma_n,stepSize)
            obj.sigma_f=sigma_f;
            obj.l=l;
            obj.sigma_n=sigma_n;
            obj.mu=stepSize;
            obj.z=[obj.sigma_f;obj.l;obj.sigma_n];
        end

        function obj = runLocalGD(obj)
            %             distX = dist((diag(obj.l)\eye(2))*obj.X).^2;%distX=(X-X^')^T Sigma^-1(X-X^')
            %             %             K_s=obj.sigma_f^(2)*exp(-0.5*distX./obj.l^(2));
            %             K_s=obj.sigma_f^(2)*exp(-0.5*distX);
            %             K_n=obj.sigma_n^2*eye(obj.N_m);
            %             obj.K=K_s+K_n;

            %             choL = chol(obj.K, 'lower');
            %             alpha = choL'\(choL\obj.Z);
            %             %             invK=inv(obj.K);
            %             invChoL=inv(choL);
            %             constant_1=invChoL'*invChoL-alpha*alpha';
            %             K_div_sigma_f=2/obj.sigma_f*K_s;
            %             %             K_div_sigma_f=2*obj.sigma_f*exp(-0.5*distX/obj.l^2);
            %             obj.pd_sigma_f = 0.5*trace(constant_1*K_div_sigma_f);
            %
            %             %             K_div_l=obj.sigma_f^2*distX*exp(-distX./2./obj.l^(2))*obj.l^(-3);
            %
            %             K_div_l_1=obj.distX1.*K_s*obj.l(1)^(-3);
            %             obj.pd_l(1) = 0.5*trace(constant_1*K_div_l_1);
            %             K_div_l_2=obj.distX2.*K_s*obj.l(2)^(-3);
            %             obj.pd_l(2) = 0.5*trace(constant_1*K_div_l_2);

            [pd,pdn] = getDiv(obj,obj.z);
            obj.pd_sigma_f=pd(1);
            obj.pd_l=pd(2:(2+length(obj.pd_l)-1));
            obj.pd_sigma_n=pdn;

            %             obj.mu=0.999999*obj.mu;
        end

        function [obj,Zs,Steps,inIterCount] = runLocalInnerADMM(obj,maxInIter,epsilon)
            epsilon=0.001;
            localGDflag=1;
            D=length(obj.l);
            %             epsilon=10*epsilon;
            inIterCount=0;
            Zs=zeros(2+D,maxInIter);
            %             old_sigma=obj.z(1);
            %             old_l=obj.z(2:3);
            old_sigma=obj.z(1);
            old_l=obj.z(2:(2+D-1));
            old_sigma_n=obj.z(end);
            % below are some unchanged matrix used for every iterations
            %             K_n=obj.sigma_n^2*eye(obj.N_m);
            mu_temp=obj.mu;
            Steps=[];
            while localGDflag
                inIterCount=inIterCount+1;
                %%%%GD Iter START%%%%
                %                 distX = dist((diag(old_l)\eye(D))*obj.X).^2;%distX=(X-X^')^T Sigma^-1(X-X^')
                %             K_s=obj.sigma_f^(2)*exp(-0.5*distX./obj.l^(2));
                %                 K_n=old_sigma_n^2*eye(obj.N_m);
                %                 K_s=old_sigma^(2)*exp(-0.5*distX);
                %                 obj.K=K_s+K_n;

                %                                 choL = chol(obj.K, 'lower');
                %                                 alpha = choL'\(choL\obj.Z);
                %                             invK=inv(obj.K);
                %                                 invChoL=inv(choL);
                %                                 constant_1=invChoL'*invChoL-alpha*alpha';
                % %                                 K_div_sigma_f=2/old_sigma*K_s;
                %                             K_div_sigma_f=2*obj.sigma_f*exp(-0.5*distX/obj.l^2);
                %                                 obj.pd_sigma_f = 0.5*trace(constant_1*K_div_sigma_f)+...
                %                                     obj.beta(1)*(1+obj.rho/obj.beta(1)*(old_sigma-obj.z(1)));
                %
                %                                 %             K_div_l=obj.sigma_f^2*distX*exp(-distX./2./obj.l^(2))*obj.l^(-3);
                %
                %                                 K_div_l_1=obj.distX1.*K_s*old_l(1)^(-3);
                %                                 obj.pd_l(1) = 0.5*trace(constant_1*K_div_l_1)+...
                %                                     obj.beta(2)*(1+obj.rho/obj.beta(2)*(old_l(1)-obj.z(2)));
                %
                %                                 K_div_l_2=obj.distX2.*K_s*old_l(2)^(-3);
                %                                 obj.pd_l(2) = 0.5*trace(constant_1*K_div_l_2)+...
                %                                     obj.beta(3)*(1+obj.rho/obj.beta(3)*(old_l(2)-obj.z(3)));
                %
                %                                 K_div_l=[obj.pd_l(1);obj.pd_l(2)];
                %                                 obj.pd_l=K_div_l;
                old_z=[old_sigma;old_l;old_sigma_n];

                [pd,pdn] = getDiv(obj,old_z);
                pd(1)=pd(1)+...
                    obj.beta(1)*(1+obj.rho/obj.beta(1)*(old_sigma-obj.z(1)));
                for i=2:(2+D-1)
                    pd(i)=pd(i)+obj.beta(i)*(1+obj.rho/obj.beta(i)*(old_l(i-1)-obj.z(i)));
                end
                pdn=pdn+obj.beta(end)*(1+obj.rho/obj.beta(end)*(old_sigma_n-obj.z(end)));



                obj.pd_l=pd(2:end);
                obj.pd_sigma_f=pd(1);
                obj.pd_sigma_n=pdn;

                % update hyperparameters
                new_sigma=old_sigma-mu_temp*obj.pd_sigma_f;
                new_l=old_l-mu_temp*obj.pd_l;
                new_sigma_n=old_sigma_n-mu_temp*pdn;

                % calculate maximum update step
                step=norm([new_sigma;new_l;new_sigma_n]-[old_sigma;old_l;old_sigma_n]);
                % now the new sigma and l will be out dated in the next
                % iteration
                old_sigma=new_sigma;
                old_l=new_l;
                old_sigma_n=new_sigma_n;

                obj.sigma_f=new_sigma;
                obj.l=new_l;
                obj.sigma_n=new_sigma_n;

                % adjust stepSize
                %                 mu_temp=0.9999999*mu_temp;
                % store intermediate results
                Zs(:,inIterCount)=[new_sigma;new_l;new_sigma_n];
                Steps=[Steps,step];
                if step<epsilon
                    localGDfalg=0;
                end
                if inIterCount>=maxInIter
                    localGDflag=0;
                end
                %%%%GD Iter END%%%%
            end
            obj.mu=mu_temp;
            obj.sigma_f=new_sigma;
            obj.l=new_l;
            obj.sigma_n=new_sigma_n;
            % after hyperparameter update, update the beta
            obj.beta=obj.beta+obj.rho*([obj.sigma_f;obj.l;obj.sigma_n]-obj.z);
            % the local update end
            Zs=Zs(:,1:inIterCount);
        end

        function obj = runLocalPxADMM(obj)
            D=length(obj.l);
            %             old_sigma=obj.z(1);
            %             old_l=obj.z(2:(2+D-1));
            old_sigma_n=obj.z(end);
            %             inputDim=length(old_l);
            %             K_n=obj.sigma_n^2*eye(obj.N_m);
            %             %%%%local update START%%%%
            %             distX = dist((diag(old_l)\eye(inputDim))*obj.X).^2;%distX=(X-X^')^T Sigma^-1(X-X^')
            %             %             K_s=obj.sigma_f^(2)*exp(-0.5*distX./obj.l^(2));
            %             K_s=old_sigma^(2)*exp(-0.5*distX);
            %             obj.K=K_s+K_n;

            % %             choL = chol(obj.K, 'lower');
            % %             alpha = choL'\(choL\obj.Z);
            %             invK=inv(obj.K);
            % %             invChoL=inv(choL);
            % %             constant_1=invChoL'*invChoL-alpha*alpha';
            % %             K_div_sigma_f=2/old_sigma*K_s;
            % %             %             K_div_sigma_f=2*obj.sigma_f*exp(-0.5*distX/obj.l^2);
            % %             obj.pd_sigma_f = 0.5*trace(constant_1*K_div_sigma_f);

            %             K_div_l=obj.sigma_f^2*distX*exp(-distX./2./obj.l^(2))*obj.l^(-3);
            % %             K_div_l=[];
            % %             for d=1:inputDim
            % %                 K_div_l_d=obj.distXd(:,:,d).*K_s*old_l(d)^(-3);
            % %
            % %                 obj.pd_l(d) = 0.5*trace(constant_1*K_div_l_d);
            % %                 K_div_l=[K_div_l;obj.pd_l(d)];
            % %             end


            %             K_div_l_1=obj.distX1.*K_s*old_l(1)^(-3);
            %             obj.pd_l(1) = 0.5*trace(constant_1*K_div_l_1);
            %
            %             K_div_l_2=obj.distX2.*K_s*old_l(2)^(-3);
            %             obj.pd_l(2) = 0.5*trace(constant_1*K_div_l_2);
            %
            %             K_div_l=[obj.pd_l(1);obj.pd_l(2)];
            %             obj.pd_l=K_div_l;
            %
            [pd,pdn] = getDiv(obj,obj.z);
            obj.pd_l=pd(2:end);
            obj.pd_sigma_f=pd(1);

            % update hyperparameters
            new_sigma=obj.z(1)-(obj.pd_sigma_f+obj.beta(1))/(obj.rho+obj.L);
            new_l=obj.z(2:(2+D-1))-(obj.pd_l+obj.beta(2:(2+D-1)))/(obj.rho+obj.L);
            new_sigma_n=old_sigma_n-(pdn+obj.beta(end))/(obj.rho+obj.L);

            obj.sigma_f=new_sigma;
            obj.l=new_l;
            obj.sigma_n=new_sigma_n;
            % after hyperparameter update, update the beta
            obj.beta=obj.beta+obj.rho*([obj.sigma_f;obj.l;obj.sigma_n]-obj.z);
            % the local update end
            [pd_new,pdn_new ]= getDiv(obj,[obj.sigma_f;obj.l;obj.sigma_n]);

            L_max=norm([pd_new;pdn_new]-[pd;pdn])/norm([obj.sigma_f;obj.l;obj.sigma_n]-obj.z);
            beta_new=obj.beta;
        end

        function obj = obtain_data_from_Neighbor_ADMM_fd(obj)
            M=length(obj);
            for m=1:M
                for n=1:obj(m).N_size
                    neighbor_idx=obj(m).Neighbors(n);
                    self_idx_at_neighbor=obj(neighbor_idx).Neighbors==m;
                    neighbour_activation_status=obj(neighbor_idx).action_status;
                    neighbour_communication_ability=obj(neighbor_idx).communicationAbility;
                    %                     obj(m).z_nm(:,n)=obj(neighbor_idx).z_mn(:,self_idx_at_neighbor);
                    if neighbour_activation_status==1||neighbour_communication_ability==1
                        obj(m).beta_nm(:,n)=obj(neighbor_idx).beta_mn(:,self_idx_at_neighbor);
                        obj(m).theta_n(:,n)=[obj(neighbor_idx).sigma_f;obj(neighbor_idx).l;obj(neighbor_idx).sigma_n];
                        obj(m).beta_n(:,n)=obj(neighbor_idx).beta;

                        obj(neighbor_idx).beta_nm(:,self_idx_at_neighbor)=obj(m).beta_mn(:,n);
                        obj(neighbor_idx).theta_n(:,self_idx_at_neighbor)=[obj(m).sigma_f;obj(m).l;obj(m).sigma_n];
                        obj(neighbor_idx).beta_n(:,self_idx_at_neighbor)=obj(m).beta;

                        obj(m).updatedVars(n)=1;
                        obj(neighbor_idx).updatedVars(self_idx_at_neighbor)=1;
                    else
                        obj(m).beta_nm(:,n)=obj(m).beta_nm(:,n);
                        obj(m).theta_n(:,n)=obj(m).theta_n(:,n);
                        obj(m).beta_n(:,n) = obj(m).beta_n(:,n);

                        obj(neighbor_idx).beta_nm(:,self_idx_at_neighbor)=obj(m).beta_mn(:,n);
                        obj(neighbor_idx).theta_n(:,self_idx_at_neighbor)=[obj(m).sigma_f;obj(m).l;obj(m).sigma_n];
                        obj(neighbor_idx).beta_n(:,self_idx_at_neighbor)=obj(m).beta;
                        obj(neighbor_idx).updatedVars(self_idx_at_neighbor)=1;
                    end
                end
            end
        end

        function [obj,Zs,Steps,inIterCount] = runLocalInnerADMM_fd(obj,maxInIter,epsilon)
            localGDflag=1;
            %             epsilon=10*epsilon;
            inIterCount=0;
            D=length(obj.l);
            Zs=zeros(D+2,maxInIter);

            old_sigma=obj.sigma_f;
            old_l=obj.l;
            old_sigma_n=obj.sigma_n;
            % below are some unchanged matrix used for every iterations
            mu_temp=obj.mu;
            Steps=[];
            for n=1:obj.N_size
                obj.z_mn(:,n) = 0.5 * (...
                    (obj.beta_mn(:,n)+obj.beta_nm(:,n))/obj.rho +...
                    ([obj.sigma_f;obj.l;obj.sigma_n] + obj.theta_n(:,n))...
                    );
            end
            while localGDflag
                inIterCount=inIterCount+1;
                %%%%GD Iter START%%%%
                [pd,pdn] = getDiv(obj,[old_sigma;old_l;old_sigma_n]);
                for n=1:obj.N_size
                    pd(1)=pd(1)+...
                        obj.beta_mn(1,n)+obj.rho*(old_sigma-obj.z_mn(1,n));
                end
                for i=2:(2+D-1)
                    for n=1:obj.N_size
                        pd(i)=pd(i)+...
                            obj.beta_mn(i,n) + obj.rho*(old_l(i-1)-obj.z_mn(i,n));
                    end
                end


                for n=1:obj.N_size
                    pdn=pdn+...
                        obj.beta_mn(end,n) + obj.rho*(old_sigma_n-obj.z_mn(end,n));
                end

                obj.pd_sigma_f=pd(1);
                obj.pd_l=pd(2:end);
                obj.pd_sigma_n=pdn;

                % update hyperparameters
                new_sigma=old_sigma-mu_temp*obj.pd_sigma_f;
                new_l=old_l-mu_temp*obj.pd_l;
                new_sigma_n=old_sigma_n-mu_temp*pdn;
                % calculate maximum update step
                step=norm([new_sigma;new_l;new_sigma_n]-[old_sigma;old_l;old_sigma_n]);
                % now the new sigma and l will be out dated in the next
                % iteration
                old_sigma=new_sigma;
                old_l=new_l;
                old_sigma_n=new_sigma_n;
                % store intermediate results
                Zs(:,inIterCount)=[new_sigma;new_l;new_sigma_n];
                Steps=[Steps,step];
                if step<epsilon
                    localGDfalg=0;
                end
                if inIterCount>=maxInIter
                    localGDflag=0;
                end
                %%%%GD Iter END%%%%
            end
            %             obj.mu=mu_temp;
            obj.sigma_f=new_sigma;
            obj.l=new_l;
            obj.sigma_n=new_sigma_n;
            obj.z=[new_sigma;new_l;new_sigma_n];
            % update z_mn and beta_mn
            for n=1:obj.N_size
                obj.beta_mn(:,n) = obj.beta_mn(:,n) + ...
                    obj.rho * ([obj.sigma_f;obj.l;obj.sigma_n]-obj.z_mn(:,n));
            end

            % the local update end
            Zs=Zs(:,1:inIterCount);
        end


        function obj=runPxADMM_fd(obj)
            D=length(obj.l);
            obj.action_status=0;
            obj.updatedVars=0*obj.updatedVars;

            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % update z_mn
            for n=1:obj.N_size
                obj.z_mn(:,n) =((obj.beta_mn(:,n))/obj.rho +([obj.sigma_f;obj.l;obj.sigma_n])+obj.beta_nm(:,n)/obj.rho+obj.theta_n(:,n))/2;
            end
            new_z=mean([obj.z,obj.z_mn],2);
%             new_z=mean([obj.z_mn],2);
%             new_z=mean([obj.beta/obj.rho+obj.theta_n],2);
            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % update theta_m
            old_sigma=new_z(1);
            old_l=new_z(2:(2+D-1));
            old_sigma_n=new_z(end);
            inputDim=length(old_l);
            obj.z=new_z;


            [pd,pdn] = getDiv(obj,obj.z);
            obj.pd_l=pd(2:end);
            obj.pd_sigma_f=pd(1);

            old_beta=mean([obj.beta,obj.beta_mn],2);

%             new_theta=new_z-([obj.pd_sigma_f;obj.pd_l;pdn]+sum([obj.beta,obj.beta_mn],2)/(obj.N_size+1))/((obj.rho+obj.L));
%             new_theta=(obj.L*obj.z-[obj.pd_sigma_f;obj.pd_l;pdn]-sum([obj.beta_mn],2)+obj.rho*sum([obj.z_mn],2))/((obj.rho*obj.N_size+obj.L));
%             new_theta=new_z-([obj.pd_sigma_f;obj.pd_l;pdn]+mean([obj.beta,obj.beta_mn],2))/((obj.rho+obj.L));
            new_theta=(1/(obj.L+obj.rho*obj.N_size))*(sum(obj.rho*obj.z_mn-obj.beta_mn,2)+obj.L*obj.z-[pd;pdn]);

            obj.sigma_f=new_theta(1);
            obj.l=new_theta(2:(2+D-1));
            obj.sigma_n=new_theta(end);
            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % beta update
            for n=1:obj.N_size
                obj.beta_mn(:,n) = obj.beta_mn(:,n) + ...
                    obj.rho * ([obj.sigma_f;obj.l;obj.sigma_n]-obj.z_mn(:,n));
            end
%             obj.beta=obj.beta+obj.rho*([obj.sigma_f;obj.l;obj.sigma_n]-new_z);
%             obj.beta=obj.beta+obj.rho*([obj.sigma_f;obj.l;obj.sigma_n]-mean(obj.z_mn,2));
            obj.beta=mean([obj.beta,obj.beta_mn],2);
        end

        function obj=runPxADMM_fd_thetac(obj)
            D=length(obj.l);
            obj.action_status=0;
            obj.updatedVars=0*obj.updatedVars;

            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % update z_mn
            for n=1:obj.N_size
                obj.z_mn(:,n) =((obj.beta_mn(:,n))/obj.rho +([obj.sigma_f;obj.l;obj.sigma_n])+obj.beta_nm(:,n)/obj.rho+obj.theta_n(:,n))/2;
            end
            new_z=mean([obj.z,obj.z_mn],2);

            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % update theta_m
            old_sigma=new_z(1);
            old_l=new_z(2:(2+D-1));
            old_sigma_n=new_z(end);
            old_theta=[obj.sigma_f;obj.l;obj.sigma_n];
            inputDim=length(old_l);
            obj.z(1)=old_sigma;
            obj.z(2:(2+D-1))=old_l;
            obj.z(end)=old_sigma_n;


            [pd,pdn] = getDiv(obj,old_theta);
            obj.pd_l=pd(2:end);
            obj.pd_sigma_f=pd(1);

            old_beta=mean([obj.beta,obj.beta_mn],2);

            new_theta=(1/(obj.L+obj.rho*obj.N_size))*(sum(obj.rho*obj.z_mn-obj.beta_mn,2)+obj.L*old_theta-[pd;pdn]);
%             new_theta=new_z-([obj.pd_sigma_f;obj.pd_l;pdn]+sum([obj.beta,obj.beta_mn],2)/(obj.N_size+1))/((obj.rho+obj.L));


            obj.sigma_f=new_theta(1);
            obj.l=new_theta(2:(2+D-1));
            obj.sigma_n=new_theta(end);
            % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
            % beta update
            for n=1:obj.N_size
                obj.beta_mn(:,n) = obj.beta_mn(:,n) + ...
                    obj.rho * ([obj.sigma_f;obj.l;obj.sigma_n]-obj.z_mn(:,n));
            end
            obj.beta=obj.beta+obj.rho*([obj.sigma_f;obj.l;obj.sigma_n]-new_z);
        end

    end
end

