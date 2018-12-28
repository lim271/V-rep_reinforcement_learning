from __future__ import print_function
import os
import sys
import time
from numpy import array,reshape,linalg,arctan2,pi,expand_dims
from random import choice
from env_modules import vrep
from env_modules.core import Core


scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'scenes')


class Robot_hand(Core):

    def __init__(self,config,port):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir,'turtlebot_obstacles.ttt')
        )
        self.goal_set = [[7.0,7.0],[7.0,7.0]]
        self.reward_param = config.reward_param
        self.action_prev = [0.0,0.0]
        self.state0 = None
        self.goal_dist_prev = None
        self.port = config.api_port
    
    def launch(self):
        self.vrep_launch()
        vrep.simxSynchronousTrigger(self.clientID)
        self.joint_handles = [
            vrep.simxGetObjectHandle(
                self.clientID,
                'Jaco_joint'+str(idx),
                vrep.simx_opmode_blocking
            )[1] for idx in range(1,7)
        ]
        self.body_handle = vrep.simxGetObjectHandle(
            self.clientID,
            'Jaco_Hand',
            vrep.simx_opmode_blocking
        )[1]
        self.brick_handle = vrep.simxGetObjectHandle(
            self.clientID,
            'Brick_red',
            vrep.simx_opmode_blocking
        )[1]
        self.epoch = 0
        self.count = 0
    
    def reset(self):
        self.vrep_reset()
        self.goal = choice(self.goal_set)
        vrep.simxSetObjectPosition(
            self.clientID,
            self.brick_handle,
            -1,
            self.goal+[0.22],
            vrep.simx_opmode_blocking
        )
        self.state0 = None
        self.action_prev = [0.0,0.0]
        self.goal_dist_prev = None
        self.epoch += self.count
        self.count = 0
        self.reward_sum = 0.0
        time.sleep(0.2)
    
    def start(self):
        self.vrep_start()
        t = vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        self.controller([0.0,0.0])
        while vrep.simxGetLastCmdTime(self.clientID)-t<self.dt:
            lrf_bin = vrep.simxGetStringSignal(
                self.clientID,
                'hokuyo_data',
                vrep.simx_opmode_streaming
            )[1]
            pose = vrep.simxGetObjectPosition(
                self.clientID,
                self.body_handle,
                -1,
                vrep.simx_opmode_streaming
            )[1]
    
    def reward(self,state,goal_dist,action):
        # return 10*(self.goal_dist_prev-goal_dist) \
        #        -(1/min(lrf)-1)/5.0 \
        #        -0.5*(1+self.reward_param*action[1]**2)
        r_g = 10.0*(self.goal_dist_prev-goal_dist)
        r_o = 0.01*sum((array(state)-array(action))**2)
        return r_g-r_o
    
    def step(self,action,return_obs = False):
        self.count += 1
        self.controller(action)
        t = vrep.simxGetLastCmdTime(self.clientID)
        vrep.simxSynchronousTrigger(self.clientID)
        while vrep.simxGetLastCmdTime(self.clientID)-t < self.dt:
            pose = vrep.simxGetObjectPosition(
                self.clientID,
                self.body_handle,
                -1,
                vrep.simx_opmode_streaming
            )[1]
            orientation = vrep.simxGetObjectOrientation(
                self.clientID,
                self.body_handle,
                -1,
                vrep.simx_opmode_streaming
            )[1][2]
            goal_pos = vrep.simxGetObjectPosition(
                self.clientID,
                self.brick_handle,
                self.body_handle,
                vrep.simx_opmode_streaming
            )[1]
            joint_pose = [
                vrep.simxGetJointPosition(
                    self.clientID,
                    self.joint_handles[idx],
                    vrep.simx_opmode_streaming
                )[1] for idx in range(6)
            ]
        goal_dist = linalg.norm(goal_pos)
        state1 = joint_pose+goal_pos
        if self.goal_dist_prev != None:
            reward = self.reward(self.state0, goal_dist, action)
            self.goal_dist_prev = goal_dist
            self.reward_sum += reward
        sys.stderr.write(
            '\rstep:%d| goal:% 2.1f,% 2.1f | pose:% 2.1f,% 2.1f | avg.reward:% 4.2f'
            %(self.count,self.goal[0],self.goal[1],pose[0],pose[1],self.reward_sum/self.count)
        )
        if goal_dist < 0.1:
            done = 1
            print(' | Success')
        else:
            done = 0
        if self.state0 != None:
            self.replay.add(
                {
                    'state0':self.state0,
                    'action0':action,
                    'reward':reward,
                    'state1':state1,
                    'done':done
                }
            )
        self.state0 = state1
        self.action_prev = action
        self.goal_dist_prev = goal_dist
        if return_obs:
            obs = {
                'state':state1,
                'action':action,
                'reward':reward
            }
            return state1,obs,done
        else:
            return state1,done
    
    def controller(self,targets):
        for idx,target in enumerate(targets):
            vrep.simxSetJointTargetPosition(
                self.clientID,
                self.joint_handles[idx],
                target,
                vrep.simx_opmode_streaming
            )