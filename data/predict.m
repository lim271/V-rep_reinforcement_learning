covfunc={'covSum',{'covSEiso','covNoise'}};
loghyper0=[0;0;log(0.0001)];

for i=1:2:9
    loghyper=minimize(loghyper0,'gpr',-100,covfunc,double(traj(i).time)/1000,traj(i).action(:,2));
    [mu S2]=gpr(loghyper,covfunc,double(traj(i).time)/1000,traj(i).action(:,2),double(traj(i).time)/1000);
    plot(double(traj(i).time)/1000,mu,double(traj(i).time)/1000,traj(i).action(:,2))
    mean(mu.^2)
    pause
end

