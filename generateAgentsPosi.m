function Agents_Posi= generateAgentsPosi(Method,range)
%GENERATEAGENTSPOSI Summary of this function goes here
%   Method: 0. totally random; 1. equal generation
range_x1=range(1,:);
range_x2=range(2,:);
scaling_factor=0.9; % prevent agents from being generated on the edge
if Method==1
    Agents_Posi=[unifrnd(range_x1(1),range_x1(2),1,maxM)*scaling_factor;
        unifrnd(range_x2(1),range_x2(2),1,maxM)*scaling_factor];
else
    
end

end

