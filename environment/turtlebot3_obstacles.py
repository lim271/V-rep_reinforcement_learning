from __future__ import print_function
import os
import sys
import time
from numpy import array, reshape, linalg, arctan2, pi, expand_dims
from numpy import max as npmax
from random import choice, uniform
from env_modules import vrep
from env_modules.core import Core


scene_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scenes')


class Turtlebot3_obstacles(Core):

    def __init__(self, config, port):
        Core.__init__(
            self,
            config,
            os.path.join(scene_dir, 'turtlebot3_obstacles.ttt')
        )
        self.d = 0.079
        self.r = 0.033
        self.target_set = [
            [ 6.0, -4.0],
            [ 6.0, -3.0],
            [ 6.0, -2.0],
            [ 6.0, -1.0],
            [ 6.0,  0.0],
            [ 6.0,  1.0],
            [ 6.0,  4.0],
            [ 6.0,  5.0],
            [ 6.0,  6.0],
            [ 5.0, -4.0],
            [ 5.0, -3.0],
            [ 5.0, -2.0],
            [ 5.0, -1.0],
            [ 5.0,  0.0],
            [ 5.0,  1.0],
            [ 5.0,  4.0],
            [ 5.0,  5.0],
            [ 5.0,  6.0],
            [ 4.0, -6.0],
            [ 4.0, -5.0],
            [ 4.0, -4.0],
            [ 4.0, -3.0],
            [ 4.0, -2.0],
            [ 4.0, -1.0],
            [ 4.0,  0.0],
            [ 4.0,  1.0],
            [ 4.0,  2.0],
            [ 4.0,  5.0],
            [ 4.0,  6.0],
            [ 3.0, -6.0],
            [ 3.0, -5.0],
            [ 3.0, -4.0],
            [ 3.0, -3.0],
            [ 3.0, -2.0],
            [ 3.0, -1.0],
            [ 3.0,  0.0],
            [ 3.0,  1.0],
            [ 3.0,  3.0],
            [ 3.0,  5.0],
            [ 3.0,  6.0],
            [ 2.0, -6.0],
            [ 2.0, -5.0],
            [ 2.0, -2.0],
            [ 2.0, -1.0],
            [ 2.0,  0.0],
            [ 2.0,  1.0],
            [ 2.0,  2.0],
            [ 2.0,  3.0],
            [ 2.0,  4.0],
            [ 2.0,  5.0],
            [ 2.0,  6.0],
            [ 1.0, -6.0],
            [ 1.0, -5.0],
            [ 1.0, -2.0],
            [ 1.0, -1.0],
            [ 1.0,  0.0],
            [ 1.0,  1.0],
            [ 1.0,  5.0],
            [ 1.0,  6.0],
            [-6.0, -4.0],
            [-6.0, -3.0],
            [-6.0, -2.0],
            [-6.0, -1.0],
            [-6.0,  0.0],
            [-6.0,  1.0],
            [-6.0,  2.0],
            [-6.0,  4.0],
            [-6.0,  5.0],
            [-6.0,  6.0],
            [-5.0, -4.0],
            [-5.0, -3.0],
            [-5.0, -2.0],
            [-5.0, -1.0],
            [-5.0,  0.0],
            [-5.0,  1.0],
            [-5.0,  2.0],
            [-5.0,  4.0],
            [-5.0,  5.0],
            [-5.0,  6.0],
            [-4.0, -6.0],
            [-4.0, -5.0],
            [-4.0, -4.0],
            [-4.0, -1.0],
            [-4.0,  2.0],
            [-4.0,  3.0],
            [-4.0,  5.0],
            [-4.0,  6.0],
            [-3.0, -6.0],
            [-3.0, -5.0],
            [-3.0, -4.0],
            [-3.0, -1.0],
            [-3.0,  0.0],
            [-3.0,  1.0],
            [-3.0,  2.0],
            [-3.0,  3.0],
            [-3.0,  5.0],
            [-3.0,  6.0],
            [-2.0, -6.0],
            [-2.0, -5.0],
            [-2.0, -4.0],
            [-2.0, -3.0],
            [-2.0, -2.0],
            [-2.0,  1.0],
            [-2.0,  2.0],
            [-2.0,  3.0],
            [-2.0,  4.0],
            [-2.0,  5.0],
            [-1.0, -6.0],
            [-1.0, -5.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [-1.0,  0.0],
            [-1.0,  1.0],
            [-1.0,  5.0],
            [ 0.0, -6.0],
            [ 0.0, -5.0],
            [ 0.0, -2.0],
            [ 0.0, -1.0],
            [ 0.0,  1.0],
            [ 0.0,  5.0],
            [ 0.0,  6.0]
        ]
        self.reward_param = config.reward_param
        self.action_prev = [0.0, 0.0]
        self.state0 = None
        self.target_dist_prev = None
        self.port = config.api_port
    
    def launch(self):
        self.vrep_launch()
        vrep.simxSynchronousTrigger(self.clientID)
        self.joint_handles = [
            vrep.simxGetObjectHandle(
                self.clientID,
                name,
                vrep.simx_opmode_blocking
            )[1] for name in ['wheel_right_joint', 'wheel_left_joint']
        ]
        self.body_handle = vrep.simxGetObjectHandle(
            self.clientID,
            'Turtlebot3',
            vrep.simx_opmode_blocking
        )[1]
        self.pose_handle = vrep.simxGetObjectHandle(
            self.clientID,
            'Turtlebot3_base',
            vrep.simx_opmode_blocking
        )[1]
        self.target_handle = vrep.simxGetObjectHandle(
            self.clientID,
            'Target',
            vrep.simx_opmode_blocking
        )[1]
        self.epoch = 0
        self.count = 0
    
    def reset(self):
        self.vrep_reset()
        self.target = choice(self.target_set)
        vrep.simxSetObjectPosition(
            self.clientID,
            self.target_handle,
            -1,
            self.target+[0],
            vrep.simx_opmode_blocking
        )
        vrep.simxSetObjectOrientation(
            self.clientID,
            self.body_handle,
            -1,
            [-pi/2, pi*uniform(-0.99, 0.99), -pi/2],
            vrep.simx_opmode_blocking
        )
        self.state0 = None
        self.action_prev = [0.0, 0.0]
        self.target_dist_prev = None
        self.epoch += self.count
        self.count = 0
        self.reward_sum = 0.0
        time.sleep(0.2)
    
    def start(self):
        self.vrep_start()
        lrf = None
        while type(lrf)==type(None):
            vrep.simxSynchronousTrigger(self.clientID)
            lrf = self.get_measurement()
        target_pos = vrep.simxGetObjectPosition(
            self.clientID,
            self.target_handle,
            self.pose_handle,
            vrep.simx_opmode_blocking
        )[1]
        self.controller([0.0, 0.0])
        target_dist = linalg.norm(target_pos[0:2])
        target_angle = arctan2(target_pos[1], target_pos[0])
        state = list(lrf)+[0.0, 0.0, target_dist/3.5, target_angle/pi]
        return state, 0
    
    def reward(self, lrf, target_dist, action):
        # return 10*(self.target_dist_prev-target_dist) \
        #        -(1/min(lrf)-1)/5.0 \
        #        -0.5*(1+self.reward_param*action[1]**2)
        r_t = 8.0*(self.target_dist_prev-target_dist)
        rho = min(lrf)
        if (rho-0.03142857)<self.reward_param/3.5:
            r_o = min((1.0/(rho-0.03142857)-3.5/self.reward_param)**2, 1)
        else:
            r_o = 0
        return r_t-r_o-0.01*action[1]**2

    def step(self, action, return_obs = False):
        self.count += 1
        self.controller(action)
        lrf = None
        while type(lrf)==type(None):
            vrep.simxSynchronousTrigger(self.clientID)
            lrf = self.get_measurement()
        pose = vrep.simxGetObjectPosition(
            self.clientID,
            self.pose_handle,
            -1,
            vrep.simx_opmode_blocking
        )[1]
        orientation = vrep.simxGetObjectOrientation(
            self.clientID,
            self.pose_handle,
            -1,
            vrep.simx_opmode_blocking
        )[1][2]
        target_pos = vrep.simxGetObjectPosition(
            self.clientID,
            self.target_handle,
            self.pose_handle,
            vrep.simx_opmode_blocking
        )[1]
        target_dist = linalg.norm(target_pos[0:2])
        target_angle = arctan2(target_pos[1], target_pos[0])
        state1 = list(lrf)+[action[0]/0.26, action[1]/0.8]+[target_dist/3.5, target_angle/pi]
        if self.target_dist_prev != None:
            reward = self.reward(lrf, target_dist, action)
            self.target_dist_prev = target_dist
            self.reward_sum += reward
        sys.stderr.write(
            '\rstep:%d| target:% 2.1f, % 2.1f | pose:% 2.1f, % 2.1f | avg.reward:% 4.2f'
            %(self.count, self.target[0], self.target[1], pose[0], pose[1], self.reward_sum/self.count)
        )
        if min(lrf) < 0.03142857:
            done = 1
            print(' | Fail')
        elif target_dist < 0.05:
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
        self.target_dist_prev = target_dist
        if return_obs:
            obs = {
                'state':state1, 
                'lidar':list(lrf), 
                'action':action, 
                'pose':[pose[0], pose[1], orientation], 
                'target':[target_dist, target_angle], 
                'reward':reward, 
                'time':t+self.dt
            }
            return state1, obs, done
        else:
            return state1, done
    
    def controller(self, action):
        vel_right = 2.0*(action[0]+self.d*action[1])/self.r
        vel_left = 2.0*(action[0]-self.d*action[1])/self.r
        vrep.simxSetJointTargetVelocity(
            self.clientID,
            self.joint_handles[0],
            vel_right,
            vrep.simx_opmode_oneshot
        )
        vrep.simxSetJointTargetVelocity(
            self.clientID,
            self.joint_handles[1],
            vel_left,
            vrep.simx_opmode_oneshot
        )

    def get_measurement(self):
        lrf_bin = vrep.simxGetStringSignal(
            self.clientID,
            'measurement',
            vrep.simx_opmode_blocking
        )[1]
        try:
            lrf = array(vrep.simxUnpackFloats(lrf_bin), dtype=float)/3.5
            lrf = npmax(reshape(lrf, [-1,10]), 1)
        except:
            lrf = None
        return lrf
