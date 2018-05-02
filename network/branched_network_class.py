import h5py 
import numpy as np
import tensorflow as tf
import os

class branched_network(object):
    """
    Tensorflow implementation of cnn from Kell et al., Neuron, 2018. 
    
    """
    def __init__(self):
        
        #Parameters
        self.rnorm_bias, self.rnorm_alpha, self.rnorm_beta = 1., 1e-3, 0.75
        self.n_labels_W = 589 ## NOTE: they're 0:587 numerically so 588 labels but add one for the genre label
        self.n_labels_G = 43 ##NOTE: they're 0:41 numerically so 42 labels but add one for the speech label 
        self.n_out_pool5_W = 6 * 6 * 512 ## NOTE: figure out how to determine this algortihmically (potentially run dummy thru?)
        self.n_out_pool5_G = 6 * 6 * 512
        
      
        #Layer Parameters
        self.layer_params_dict = {
                'data':{'edge': 256},
                'conv1': {'edge': 9, 'stride': 3, 'n_filters': 96},
                'rnorm1': {'radius': 2}, 
                'pool1': {'edge': 3, 'stride': 2},
                'conv2': {'edge': 5, 'stride': 2, 'n_filters': 256},
                'rnorm2': {'radius': 2}, 
                'pool2': {'edge': 3, 'stride': 2},
                'conv3': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv4_W': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv4_G': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv5_W': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv5_G': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'pool5_W': {'edge': 3, 'stride': 2},
                'pool5_G': {'edge': 3, 'stride': 2},
                'fc6_W': {'n_units': 1024},
                'fc6_G': {'n_units': 1024},
                'fctop_W': {'n_units': self.n_labels_W},
                'fctop_G': {'n_units': self.n_labels_G}
        }
        
        
        # Placeholders for input and output
        self.x = tf.placeholder(tf.float32, shape=[None, 
                                self.layer_params_dict['data']['edge']*self.layer_params_dict['data']['edge']])
        
        #Load saved weights and biases- split into three .npy files to upload to github
        weights_biases = np.load(os.getcwd()+ '/network/weights/network_weights_early_layers.npy')[()]
        genre_branch =  np.load(os.getcwd()+ '/network/weights/network_weights_genre_branch.npy')[()]
        word_branch = np.load(os.getcwd()+ '/network/weights/network_weights_word_branch.npy')[()] 
        weights_biases.update(genre_branch)
        weights_biases.update(word_branch)
        self.layer_vars_dict = weights_biases

        #Make the graph
        self.session, self.word_logits, self.genre_logits = self.get_graph()
    
    
    #Functions for making graph layers 
    def conv_layer(self, previous_h, layer_params, layer_vars):
        return tf.nn.relu(tf.nn.conv2d( previous_h, layer_vars['W'], 
                                  strides=[1,layer_params['stride'], layer_params['stride'],1],
                              padding='SAME' ) + layer_vars['b'])
 

    def lrnorm_layer(self, previous_h, layer_params):
        return tf.nn.local_response_normalization(   previous_h,
                                                     depth_radius = layer_params['radius'],
                                                     bias = self.rnorm_bias,
                                                     alpha = self.rnorm_alpha,
                                                     beta = self.rnorm_beta )


    def pool_layer(self, previous_h, layer_params, max_pool = True):
        if max_pool: 
            return tf.nn.max_pool(  previous_h,
                                    ksize=[1,int(layer_params['edge']),int(layer_params['edge']),1],
                                    strides=[1,int(layer_params['stride']),int(layer_params['stride']),1],
                                    padding='SAME' )
        
                            
        elif not max_pool:
            return tf.nn.avg_pool(  previous_h, 
                                    ksize=[1, layer_params['edge'], layer_params['edge'],1],
                                    strides=[1, layer_params['stride'], layer_params['stride'],1],
                                    padding='SAME' )

    def flatten_pool_layer(self, previous_layer, output_size): 
        return tf.reshape(previous_layer, [-1, output_size])
    
    def get_graph(self):
    
        #Shared layers
        x_reshape = tf.reshape(self.x, [-1, self.layer_params_dict['data']['edge'], 
                                        self.layer_params_dict['data']['edge'],1])
        h_conv1 = self.conv_layer(x_reshape, self.layer_params_dict['conv1'], self.layer_vars_dict['conv1'])
        h_rnorm1 = self.lrnorm_layer(h_conv1, self.layer_params_dict['rnorm1'])
        h_pool1 = self.pool_layer( h_rnorm1, self.layer_params_dict['pool1'])
        h_conv2 = self.conv_layer(h_pool1, self.layer_params_dict['conv2'], self.layer_vars_dict['conv2'])
        h_rnorm2 = self.lrnorm_layer( h_conv2, self.layer_params_dict['rnorm2'])
        h_pool2 = self.pool_layer( h_rnorm2, self.layer_params_dict['pool2'])
        h_conv3 = self.conv_layer(h_pool2, self.layer_params_dict['conv3'], self.layer_vars_dict['conv3'])

        #Speech Branch 
        h_conv4_W = self.conv_layer(h_conv3, self.layer_params_dict['conv4_W'], self.layer_vars_dict['conv4_W'])
        h_conv5_W = self.conv_layer(h_conv4_W, self.layer_params_dict['conv5_W'], self.layer_vars_dict['conv5_W'])
        h_pool5_W = self.pool_layer( h_conv5_W, self.layer_params_dict['pool5_W'], False)
        h_pool5_flat_W  = self.flatten_pool_layer( h_pool5_W , self.n_out_pool5_W )
        h_fc6_W = tf.nn.relu(tf.matmul(h_pool5_flat_W, 
                                       self.layer_vars_dict['fc6_W']['W']) + self.layer_vars_dict['fc6_W']['b']) 
        y_conv_logits_W = tf.matmul(h_fc6_W, 
                                    self.layer_vars_dict['fctop_W']['W']) + self.layer_vars_dict['fctop_W']['b']

        #Genre Branch 
        h_conv4_G = self.conv_layer(h_conv3, self.layer_params_dict['conv4_G'], self.layer_vars_dict['conv4_G'])
        h_conv5_G = self.conv_layer(h_conv4_G, self.layer_params_dict['conv5_G'], self.layer_vars_dict['conv5_G'])
        h_pool5_G = self.pool_layer( h_conv5_G, self.layer_params_dict['pool5_G'], False)
        h_pool5_flat_G = self.flatten_pool_layer(h_pool5_G, self.n_out_pool5_G)
        h_fc6_G= tf.nn.relu(tf.matmul(h_pool5_flat_G, 
                                      self.layer_vars_dict['fc6_G']['W']) + self.layer_vars_dict['fc6_G']['b']) 
        y_conv_logits_G = tf.matmul(h_fc6_G, 
                                    self.layer_vars_dict['fctop_G']['W']) + self.layer_vars_dict['fctop_G']['b'] 
        #Make tf session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        return session, y_conv_logits_W, y_conv_logits_G
