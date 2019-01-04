class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.gpu = True
        self.state_dim = 40
        self.action_dim = 2
        self.action_bounds = [[0.22, 2.00], [0.0, -2.00]] # [max, min]
        self.gamma = 0.9 # discount factor
        self.layers = [256, 256, 128] # [hidden1, hidden2, ... ]
        self.critic_learning_rate = 1e-4
        self.actor_learning_rate = 1e-3
        self.tau = 1e-3
        self.l2_penalty = 1e-4
        self.max_buffer = 1e+5
        self.batch_size = 128
        self.max_step = 500
        self.max_episode = 3000
        self.max_epoch = 1000000
        self.reward_param = 0.0
        self.vrep_path = '/opt/vrep'
        self.autolaunch = True
        self.visualization = True
        self.dt = 100 # milisecond
        self.api_port = 20000
        

config = Settings()
