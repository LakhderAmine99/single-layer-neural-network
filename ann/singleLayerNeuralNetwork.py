import numpy as np

class SingleLayerNeuralNetwork(object):
    
    def __init__(self,iterations=1,activation_function=None,weights_interval=[0,1],biais=0.5,alpha=0.1):
        self.iterations = iterations
        self.activation_function = activation_function
        self.low = weights_interval[0]
        self.high = weights_interval[1]
        
        self.alpha = alpha
        self.biais = biais
        
    def _eval_(self,X,w):
        return np.sum(np.multiply(X,w)) + self.biais
    
    def _adjust_(self,X,y,output):    
        self.weights += self.alpha*(y-output)*X
        self.biais += self.alpha*(y-output)
            
    def _feed_forward_(self,X,w):
        return self.activation_function(self._eval_(X,w))
    
    def _SLNN_(self,X,y):
        
        self.output = np.zeros(shape=(X.shape[0]))
        self.weights = np.random.randint(low=self.low*10,high=self.high*10,size=(X.shape[1]))/10

        while self.iterations > 0:
            
            for i in range(X.shape[0]):

                self.output[i] = self._feed_forward_(X.iloc[i],self.weights)
                error = y[i] - self.output[i]
                
                if error != 0:
                    self._back_propagation_(X.iloc[i],y.iloc[i],self.output[i])
                    
            self.iterations -= 1
        
        return self.output
    
    def _back_propagation_(self,X,y,output):  
        self._adjust_(X,y,output)    
        
    def weights(self):
        return self.weights
    
    def fit(self,X,y):
        return self._SLNN_(X,y)
            
    def predict(self,X):
        
        self.predictions = np.zeros(shape=(X.shape[0]))
                
        for i in range(X.shape[0]):
            self.predictions[i] = self._feed_forward_(X.iloc[i],self.weights)
        
        return self.predictions
    