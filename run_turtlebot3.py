from __future__ import print_function
import os, sys, time
from numpy import reshape, save, load
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot3_obstacles import Turtlebot3_obstacles

# config.autolaunch = False
config.reward_param = 0.4

def train(port):    
    config.api_port = port
    env = Turtlebot3_obstacles(config, port)

    agent = DDPG(config)
    # agent.load(load('savedir/weight.npy').item())

    env.launch()

    env.epoch = 0
    env.reward_param = param
    for episode in range(config.max_episode):
        env.reset()
        print('Rho:', param, '\tEpisode:', episode+1)
        state, done = env.start()
        for step in range(config.max_step):
            epsilon = 0.3*(0.99999**env.epoch)
            action = agent.policy(reshape(state, [1, config.state_dim]), epsilon=epsilon)
            state, done = env.step(reshape(action, [config.action_dim]))
            if env.replay.buffersize > 256:
                batch = env.replay.batch()            
                agent.update(batch)
            if done == 1:
                break
        if step >= config.max_step-1:
            print(' | Timeout')
        if (episode+1)%100 == 0:
            save(
                os.path.join(
                    'savedir',
                    'weight.npy'
                ),
                agent.return_variables()
            )
        if env.epoch > config.max_epoch:
            break

def test(port):
    config.api_port = port
    env = Turtlebot3_obstacles(config, port)
    agent = DDPG(config)

    env.launch()

    env.reward_param = 0.4
    agent.load(load('savedir/weight.npy').item())
    env.reset()
    env.start()
    state, done = env.step([0, 0])
    traj = []
    for step in range(config.max_step):
        action = agent.policy(reshape(state, [1, config.state_dim]), epsilon=0.0)
        state, obs, done = env.step(reshape(action, [config.action_dim]), return_obs=True)
        traj.append(obs)
        if done == 1:
            break
    if step >= config.max_step-1:
        print(' | Timeout')
    save('recovered_trajectory.npy', traj)

if __name__ == '__main__':
    train(19999)
    #test(20000)
