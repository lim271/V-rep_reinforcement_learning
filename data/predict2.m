X=[]
y=[]
for i=1:2:9
    X=[X;...
        traj(i).lidar ...
        traj(i).goal ...
        ones(length(traj(i).time),1)*traj(i).param];
    y=[y;traj(i).action(:,2)];
end

loghyper0=zeros(41,1);
loghyper0(end)=log(0.2);
covfunc={'covSum',{'covSEard','covNoise'}}

loghyper=minimize(loghyper0,'gpr',-5000,covfunc,X,y);

[mu S2]=gpr(loghyper,covfunc,X,y,X);