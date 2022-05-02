function Agents = runPXADMM_fd_spmd(Agents,epsilon)

M=length(Agents);
q = parallel.pool.DataQueue; % define a dataReceiver in center to monitor process in agents
afterEach(q,@tackleData)
warning('off','all')
tic
spmd

    warning('off','all')
    m=labindex;
    maxIter=Agents(m).maxIter;
    iterCount=0;
    iterFlag=1;
    updateFlag=1;

    updateCountDown=5;
    while iterCount<maxIter

        iterCount=iterCount+1;
        %         Agents(m).z=Agents(m).z.*rand;
        %         Agents(m).beta=Agents(m).beta .* (rand*rand);
        msgSentPkg=cell(7,1);
        msgSentPkg{1}=Agents(m).Code;
        msgSentPkg{3}=Agents(m).beta;
        msgSentPkg{5}=[Agents(m).sigma_f;Agents(m).l];
        msgSentPkg{6}=Agents(m).currentStep;
        msgSentPkg{7}=iterCount;
        cStep=0;
        cStepCount=0;
        for srcWkrIdx=Agents(m).Neighbors
            rcvWkrIdx=srcWkrIdx;
            NeighborIdx=find(Agents(m).Neighbors==rcvWkrIdx);
            msgSentPkg{4}=Agents(m).beta_mn(:,NeighborIdx);
            msgSentPkg{2}=srcWkrIdx;

            labSend(msgSentPkg,rcvWkrIdx);
            %             maxWait=160;
            %             isDataAvail = labProbe(srcWkrIdx);
            %             while (~isDataAvail)&&(maxWait>0)
            %                 pause(0.0001)
            %                 isDataAvail = labProbe(srcWkrIdx);
            %                 maxWait=maxWait-1;
            %             end
            %             if isDataAvail
            %                 cStep=[cStep,msgSentPkg{6}];
            %                 msgReceivedPkg=labReceive(srcWkrIdx);
            %                 neighborCode=msgReceivedPkg{1};
            %                 NeighborIdx=find(Agents(m).Neighbors==neighborCode);
            %                 Agents(m).beta_n(:,NeighborIdx)=msgReceivedPkg{3};
            %                 Agents(m).beta_nm(:,NeighborIdx)=msgReceivedPkg{4};
            %                 Agents(m).theta_n(:,NeighborIdx)=msgReceivedPkg{5};
            %                 %                 leastUpdateNum=leastUpdateNum-1;
            %                 %                 send(q,msgReceivedPkg);
            %             end
        end



        Neighbors=Agents(m).Neighbors;
        NotNeighbors=1:1:M;
        NotNeighbors(Neighbors)=[];
        Agents(m).slowSync(NotNeighbors)=-1;
        Agents(m).slowSync(Neighbors)=Agents(m).slowSync(Neighbors)+1;
        mustUpdate=zeros(M,1);
        mustUpdate(Agents(m).slowSync>=Agents(m).slowSyncThreshold)=1;
        mustUpdateNum=sum(mustUpdate);
        N_size=Agents(m).N_size;
        restNum=N_size;

        msgReceivedPkgs=cell(N_size,1);

        outMaxIter=300;
        Agents(m).updatedVarsNumber=0;
        NodeSelector=1;

        while ((Agents(m).updatedVarsNumber<Agents(m).updatedVarsNumberThreshold)&&(outMaxIter>0))||(mustUpdateNum>0)
            %             commuIdx=mod(outMaxIter,N_size)+1;
            commuIdx=mod(NodeSelector,N_size)+1;

            srcWkrIdx=Neighbors(commuIdx);
            if srcWkrIdx~=0
                outMaxIter=outMaxIter-1;
                inMaxIter=10;
                isDataAvail = labProbe(srcWkrIdx);
                while (~isDataAvail)&&(inMaxIter>0)
                    pause(0.00005)
                    isDataAvail = labProbe(srcWkrIdx);
                    inMaxIter=inMaxIter-1;
                end
		if (~((Agents(m).updatedVarsNumber<Agents(m).updatedVarsNumberThreshold)&&(outMaxIter>0)))&&(~isDataAvail)
	   rcvWkrIdx=srcWkrIdx;
	  NeighborIdx=find(Agents(m).Neighbors==rcvWkrIdx);
	   msgSentPkg{4}=Agents(m).beta_mn(:,NeighborIdx);
	   msgSentPkg{2}=srcWkrIdx;

	   labSend(msgSentPkg,rcvWkrIdx);
		end

                if isDataAvail
                    msgReceivedPkg=labReceive(srcWkrIdx);
                    Agents(m).updatedVarsNumber=Agents(m).updatedVarsNumber+1;
                    restNum=restNum-1;
                    msgReceivedPkgs{commuIdx}=msgReceivedPkg;
                    Neighbors(commuIdx)=0;
                    Agents(m).slowSync(srcWkrIdx)=0;
                    mustUpdate(srcWkrIdx)=0;
                    mustUpdateNum=sum(mustUpdate);
                end
                NodeSelector=NodeSelector+1;
            else
                outMaxIter=outMaxIter;
                NodeSelector=NodeSelector+1;
            end

            if restNum==0 % all data transfer down, force to break
                break;
            end

        end
        neighborMaxIterCount=max(0,iterCount);
        for n=1:N_size
            if Neighbors(n)==0
                msgReceivedPkg=msgReceivedPkgs{n};
                neighborCode=msgReceivedPkg{1};
                NeighborIdx=find(Agents(m).Neighbors==neighborCode);
                Agents(m).beta_n(:,NeighborIdx)=msgReceivedPkg{3};
                Agents(m).beta_nm(:,NeighborIdx)=msgReceivedPkg{4};
                Agents(m).theta_n(:,NeighborIdx)=msgReceivedPkg{5};
                cStep=cStep+msgReceivedPkg{6};
                cStepCount=cStepCount+1;
                neighborMaxIterCount=max(neighborMaxIterCount,msgReceivedPkg{7});
            end
        end
        iterCount=neighborMaxIterCount;




        if ~isempty(cStep)
            cStep(1)=[];
            cStep=mean(cStep);
        end

        if updateFlag==1
            Agents(m)=Agents(m).runPxADMM_fd;
            new_z=[Agents(m).sigma_f;Agents(m).l];
            old_z=Agents(m).Zs(:,iterCount);
            Agents(m).Zs=[Agents(m).Zs,[Agents(m).sigma_f;Agents(m).l]];
            step=max(vecnorm(new_z(1:end)-old_z(1:end),2,2));
            Agents(m).Steps=[Agents(m).Steps,step];
            Agents(m).currentStep=step;
        else
            step=Agents(m).Steps(end);
            Agents(m).Steps=[Agents(m).Steps,step];
            Agents(m).currentStep=step;
            updateCountDown=updateCountDown-1;
            new_z=[Agents(m).sigma_f;Agents(m).l];
            old_z=Agents(m).Zs(:,end);
            Agents(m).Zs=[Agents(m).Zs,[Agents(m).sigma_f;Agents(m).l]];
        end

        if  (1/(cStepCount+1)*step+cStepCount/(cStepCount+1)*cStep)<epsilon
            updateFlag=1;
        end
        if updateCountDown==0
            updateCountDown=10;
            updateFlag=1;
        end

    end

end
newAgents=agent.empty(M,0);
for m=1:M
    am=Agents(m);
    am=am{1};
    newAgents(m)=am(m);
end
Agents=newAgents;

toc

end


function tackleData(q)
fprintf("%s %d %s %d \n","From agent ",q{1},"received information from agent ",q{2});
disp("received z")
disp(q{3})
disp("received beta")
disp(q{4})
end
