from __future__ import print_function
import os,sys,time
from numpy import reshape,save,load
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot_obstacles import Turtlebot_obstacles

config.reward_param=0.8

# env.launch()

def train(port):
    config.api_port=port
    config.state_dim=9
    config.action_bounds=[
        [],
        []
    ]
    config.action_dim=6
    env=Turtlebot_obstacles(config,port)

    agent=DDPG(config)
    # agent.load(load('savedir/weight_0.0.npy').item())

    env.launch()

    for episode in range(config.max_episode):
        env.reset()
        print('Episode:',episode+1)
        env.start()
        state,done=env.step([0,0])
        for step in range(config.max_step):
            epsilon=0.99999**env.epoch
            action=agent.policy(reshape(state,[1,config.state_dim]),epsilon=epsilon)
            state,done=env.step(reshape(action,[config.action_dim]))
            if env.replay.buffersize>100:
                batch=env.replay.batch()            
                agent.update(batch)
            if done==1:
                break
        if step>=config.max_step-1:
            print(' | Timeout')
        if (episode+1)%100==0:
            save(os.path.join( \
                'savedir','weight_'+str(config.reward_param)+'.npy'), \
                agent.return_variables())
        if env.epoch>config.max_epoch:
            break

if __name__=='__main__':
    train(20000)