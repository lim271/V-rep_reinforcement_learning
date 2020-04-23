import numpy as np
import matplotlib.pyplot as plt


data = np.load('result_trajectories.npy', allow_pickle=True)

trajs=[]
for item in data:
    traj=[]
    for dct in item:
        traj.append(dct['pose'][0:2])
    trajs.append(np.array(traj))

for traj in trajs:
    n = len(traj)
    lb=np.random.randint(0,int(n/2))
    ub=np.random.randint(int(n/2),n)
    plt.plot(traj[lb:ub,0],traj[lb:ub,1])

plt.axis('equal')

plt.show()
