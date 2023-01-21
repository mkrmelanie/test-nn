# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:51:17 2023

@author: Rittenhouse
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i +1] + 1)
            self.W.append(w/np.sqrt(layers[i]))
            
            w = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(w / np.sqrt(layers[-2]))
            
            def activate(self, x):
                return 1.0 / (1 + np.exp(-x))
            
            def dactivate(self, x):
                return x * (1-x)
            
            def fitpartial(self, x, y):
                A = [np.atleast_2d(x)]
                
                for layer in np.arange(0, len(self.W)):
                    net = A[layer].dot(self.W[layer])
            #feedforward part: dot product between A and W
                    out = self.activate(net)
                    A.append(out)
            #net output added onto A
            
                error = A[-1] - y
                D = [error * self.dactivate(A[-1])]
                
                for layer in np.arange(len(A) - 2, 0, -1):
                    delta = D[-1].dot(self.W[layer].T)
                    delta = delta * self.dactivate(A[layer])
                    D.append(delta)
                    
                    D = D[::-1]
                        #reverse deltas because we went through list in reverse
                        #weight updates: dot product btwn A and D then multiply by learning rate, add to W
                    for layer in np.arange(0, len(self.W)):
                        self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])