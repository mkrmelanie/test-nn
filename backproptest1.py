# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:47:02 2023

@author: Rittenhouse
"""

import numpy as np

class NeuralNetwork:
	def __init__(self, layers, alpha=0.1):
		self.W = [] #makes list of weights
		self.layers = layers #stores layers
		self.alpha = alpha #stores alpha
#layers = describes the # of nodes in each layer
#alpha = learning rate 

        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i +1] + 1) 
            self.W.append(w/np.sqrt(layers[i]))
#make a random MxN weight mx and then connect layers together, add an extra layer for bias

		w = np.random.randn(layers[-2] + 1, layers[-1])
		self.W.append(w / np.sqrt(layers[-2]))
#special case where input layers need bias and output does not


    def activate(self, x):
        return 1.0 / (1 + np.exp(-x))
    #sigmoid activation function
    
    def dactivate(self, x):
        return x * (1-x)
    #derivative of activ. function
    
    
    
    
    #insert training stuff here
    
    
    
    
    
    #******PRIMARY BACKPROP PART HERE********:
    def fitpartial(self, x, y):
        A = [np.atleast_2d(x)]
    #as data pt x moves through the network, list A stores activation outputs in each layer
        
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
    #feedforward part: dot product between A and W
            A.append(out)
    #net output added onto A
        
        
    #backward pass part: start by getting error (difference between prediction and actual)
        error = A[-1] - y
        D = [error * self.dactivate(A[-1])]
    #use chain rule to find deltas, which will update A
    #chain rule loop:
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.dactivate(A[layer])
			D.append(delta)
            
        D = D[::-1]
    #reverse deltas because we went through list in reverse
    #weight updates: dot product btwn A and D then multiply by learning rate, add to W
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
#https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
            
    