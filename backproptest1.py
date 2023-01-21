# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:51:17 2023

@author: Rittenhouse
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):  #create weight matrices, store layers and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i +1] + 1)    #make a random weight mx connecting all nodes, extra node for bias
            self.W.append(w/np.sqrt(layers[i]))
            
            w = np.random.randn(layers[-2] + 1, layers[-1])     #special case for last 2 layers where input needs bias and output doesn't
            self.W.append(w / np.sqrt(layers[-2]))
            
            def activate(self, x):
                return 1.0 / (1 + np.exp(-x))   #sigmoid activation function
            
            def dactivate(self, x):
                return x * (1-x)    #derivative of sigmoid activ. function
            
            #///////////////insert training stuff here////////////////////////
            
            def fitpartial(self, x, y):
                A = [np.atleast_2d(x)]  #make list of output activations for each layer
                
                for layer in np.arange(0, len(self.W)): #FEEDFORWARD
                    net = A[layer].dot(self.W[layer])   #feedforward activation through the dot product of activation and weight matrices
                    out = self.activate(net)    #creates net input
                    A.append(out)   #add to activation list
            
                error = A[-1] - y   #BACKPROP
                D = [error * self.dactivate(A[-1])]     #error = final output in activation - target value, then use chain rule to build list of deltas
                
                for layer in np.arange(len(A) - 2, 0, -1): #loop in reverse order to work backwards through network
                    delta = D[-1].dot(self.W[layer].T)
                    delta = delta * self.dactivate(A[layer])
                    D.append(delta)
                    
                    D = D[::-1] #reverse deltas
                    for layer in np.arange(0, len(self.W)): #weight update phase
                        self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
                        
                        
#https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
