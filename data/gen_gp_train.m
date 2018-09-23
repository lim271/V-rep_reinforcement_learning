load('traj.mat');
X=[];
y=[];
for idx=1:2:9
    X=[X;min(traj(idx).lidar(:)) ...
        mean(traj(idx).lidar(:)) ...
        mean(min(traj(idx).lidar,[],2))];
    y=[y;traj(idx).param];
end