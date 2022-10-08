from random import randint, random
from turtle import shape
import numpy as np

class SingleLayerNeuralNetwork(object):
    
    def __init__(self,iterations=1,activation_function=None,weights_interval=[-0.5,0.5]):
        self.iterations = iterations
        self.activation_function = activation_function
        self.low = weights_interval[0]
        self.high = weights_interval[1]
        
    def _eval_(self,X,w):
        return np.sum(np.multiply(X,w))
    
    def _adjust_(self):
        return
    
    def _feed_forward_(self,X,w):
        return self.activation_function(self._eval(X,w))
    
    def _SLNN_(self,X,y):
        
        self.errors = np.zeros(shape=(X.shape[0]))
        self.output = np.zeros(shape=(X.shape[0]))
        self.weights = np.random.randint(low=self.low*10,high=self.high*10,size=(X.shape[1]))/10

        while self.iterations > 0:
            
            for i in range(X.shape[0]):

                self.output[i] = self._feed_forward_(X.iloc[i],self.weights)
                self.errors[i] = y[i] - self.output[i]
                
                if self.errors[i] != 0:
                    self.weights = self._back_propagation_(X.iloc[i],y.iloc[i],self.output[i])
                    
            self.iterations -= 1
        
        return self.output
    
    def _back_propagation_(self,X,y,output):
        
        # update the weights
        
        self._adjust_()
        
        return       
    
    def fit(self,X,y):
        return self._SLNN_(X,y)
            
    def predict(self,X):
        return self._feed_forward_(X,self.weights)
    