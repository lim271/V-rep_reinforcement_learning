from __future__ import print_function
import os, sys, time
from numpy import reshape, save, load
from configuration import config

from agent.ddpg import DDPG
from environment.turtlebot3_obstacles import Turtlebot3_obstacles

config.reward_param = 0.4

def train(port):    
    config.api_port = port
    env = Turtlebot3_obstacles(config, port)

    agent = DDPG(config)
    # agent.load(load('savedir/weight.npy').item())

    env.launch()

    env.epoch = 0
    env.reward_param = config.reward_param
    for episode in range(config.max_episode):
        env.reset()
        print('Episode:', episode+1)
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

    env.reward_param = 0.2576
    agent.load(load('savedir/weight_'+str(env.reward_param)+'.npy', allow_pickle=True, encoding='bytes').item())
    # agent.load(load('savedir/weight.npy').item())
    success_count = 0
    fail_count = 0
    timeout_count = 0
    trajs = []
    for episode in range(1000):
        env.reset()
        env.start()
        state, done = env.step([0, 0])
        traj = []
        for step in range(config.max_step):
            action = agent.policy(reshape(state, [1, config.state_dim]), epsilon=0.0)
            state, obs, done = env.step(reshape(action, [config.action_dim]), return_obs=True)
            traj.append(obs)
            if done == 1:
                if obs['success']:
                    success_count += 1
                else:
                    fail_count += 1
                break
        if step >= config.max_step-1:
            timeout_count += 1
            print(' | Timeout')
        trajs.append(traj)
        # save('recovered_trajectory.npy', traj)
    save('obstacle_avoid_result.npy', [success_count, fail_count, timeout_count])
    save('result_trajectories.npy', trajs, allow_pickle=True)

if __name__ == '__main__':
    #train(19999)
    test(20000)
