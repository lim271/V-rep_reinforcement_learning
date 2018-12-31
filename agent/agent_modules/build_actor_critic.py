import tensorflow as tf
import math

class Build_network(object):

    def __init__(self, sess, config, name):
        self.name = name
        self.sess = sess
        self.trainable = False if name.split('_')[1] == 'target' else True
        with tf.name_scope(name):
            self.state = tf.placeholder(tf.float32, [None, config.state_dim])
            layers = [[config.state_dim, config.layers[0]]]
            for layer in zip(config.layers[:-1], config.layers[1:]):
                layers.append(list(layer))
            if name[0] == 'a':
                layers.append([config.layers[-1], config.action_dim])
                self.a_scale = tf.subtract(
                    config.action_bounds[0], config.action_bounds[1])/2.0
                self.a_mean = tf.add(
                    config.action_bounds[0], config.action_bounds[1])/2.0
            else:
                layers.append([config.layers[-1], 1])
                layers[0][0] += config.action_dim
                self.action = tf.placeholder(tf.float32, [None, config.action_dim])
            for idx, (in_dim, out_dim) in enumerate(layers):
                self.create_variable(in_dim, out_dim, 'fc'+str(idx))
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
            self.variables = {var.name:var for var in self.var_list}
            out_ = tf.concat([self.state, self.action], 1) if name[0] == 'c' else self.state
            for layer in range(len(layers)-1):
                out_ = tf.nn.leaky_relu(self.layer_fc(out_, 'fc'+str(layer)))
            out_ = self.layer_fc(out_, 'fc'+str(len(layers)-1))
            if name[0] == 'a':
                self.out_ = tf.multiply(tf.tanh(out_), self.a_scale)+self.a_mean
            else:
                self.out_ = out_

    def evaluate(self, state, action = None):
        if self.name[0] == 'a':
            feed_dict = {self.state:state}
        else:
            feed_dict = {self.state:state, self.action:action}
        return self.sess.run(self.out_, feed_dict=feed_dict)    

    def layer_fc(self, in_, layer):
        return tf.matmul(in_,
            self.variables[self.name+'/'+layer+'/w:0']
        )+self.variables[self.name+'/'+layer+'/b:0']    

    def create_variable(self, in_dim, out_dim, name):
        with tf.name_scope(name):
            tf.Variable(
                tf.random_uniform(
                    [in_dim, out_dim],
                    -1/math.sqrt(in_dim),
                    1/math.sqrt(in_dim)
                ),
                name='w',
                trainable=self.trainable
            )
            tf.Variable(
                tf.random_uniform(
                    [out_dim],
                    -1/math.sqrt(in_dim),
                    1/math.sqrt(in_dim)
                ),
                name='b',
                trainable=self.trainable
            )