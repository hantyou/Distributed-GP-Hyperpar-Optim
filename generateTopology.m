function [A_full,Agents]=generateTopology(Agents,methods)


M=Agents(1).M;
if methods==1
    %% method 1: stacking squares
    A4=zeros(4,4);
    A4(1,2)=1;A4(1,4)=1;
    A4(2,3)=1;A4(3,4)=1;
    A4=A4+A4';
    blockNum=ceil(M/4);
    A_size=4*blockNum;
    A_full=zeros(A_size);

    for l=1:blockNum
        start_i=(l-1)*4+1;
        start_j=start_i;
        A_full(start_i:start_i+3,start_j:start_j+3)=A4;
        if l>=2
            A_full(start_i:start_i+3,start_j-4:start_j-1)=eye(4);
            A_full(start_i-4:start_i-1,start_j:start_j+3)=eye(4);
        end
    end
    A_full=A_full(1:M,1:M);
elseif methods==2
    %% method 2: nearest link with minimum link
    Agent_1_posi=Agents(1).Position;
    Posi_dim=length(Agent_1_posi);
    Agents_posi=zeros(Posi_dim,M);
    for m=1:M
        Agents_posi(:,m)=Agents(m).Position;
    end
    Network_dist=dist(Agents_posi);
    A_full=zeros(M,M);
    for m=1:M
        A_full_m=zeros(1,M);
        dist_m=Network_dist(m,:);
        dist_m(m)=Inf;
        A_full_m(dist_m<=Agents(m).commuRange)=1;
        if sum(A_full_m)==0
            A_full_m(dist_m==min(dist_m))=1;
        end
        A_full(m,:)=A_full(m,:)|A_full_m;
        A_full(:,m)=A_full(:,m)|(A_full_m');
    end
elseif methods==3
    %% method 2: no links
    A_full=zeros(M,M);
end

for m=1:M
    ind_m=find(A_full(m,:)~=0);
    Agents(m).Neighbors=ind_m;
    Agents(m).A=A_full;
end

if nargout==2
    disp('Neighbor info saved into agents.')
end



end
