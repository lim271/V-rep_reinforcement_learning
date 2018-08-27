import numpy as np
from scipy import io as sio
import copy

data=np.load('recorded_traj.npy').item()

for param in data.keys():
    mat={}
    for key in data[param][0].keys():
        tmp=[]
        for dct in data[param]:
            tmp.append(dct[key])
        mat[key]=copy.copy(tmp)
    sio.savemat(param+'.mat',mat)
