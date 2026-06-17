import numpy as np
import pickle

class LinearRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.weights=None
        self.bias=None
        self.learning_rate=learning_rate
        self.iterations=iterations
    def gradient(self,X,y,y_predicted):
        m=len(y_predicted)
        self.dW=np.dot(X.T,(y_predicted-y))/m
        self.db=np.sum(y_predicted-y)/m
        pass

    def error_func(self,y_predicted,y):
        m=len(y_predicted)
        cost=np.sum((y_predicted-y)**2/(2*m))
        return cost
        

    def fit(self,X,y): 
        x_converted=np.array(X)
        y_converted=np.array(y)
        N_samples,N_features=x_converted.shape
        self.weights=np.zeros(N_features)
        self.bias=0.0
        for _ in range(self.iterations):
          y_predicted=self.predict(x_converted)
          self.gradient(x_converted,y_converted,y_predicted)
          self.weights=self.weights-(self.dW*self.learning_rate)
          self.bias=self.bias-(self.db*self.learning_rate)

    def save_model(self,filename=None):


        with open (filename,"wb") as file:
            model_info={
                "weights":self.weights,
                "bias":self.bias,
                "learning_rate":self.learning_rate
            }
            pickle.dump(model_info,file)
    @classmethod
    def load_model(cls,filename):
        with open (filename,"rb") as file:
            model_info=pickle.load(file)
        model=cls(learning_rate=model_info["learning_rate"])
        model.weights=model_info["weights"]
        model.bias=model_info["bias"]
        return model


    
    def predict(self,X):
        return np.dot(X, self.weights) + self.bias
    def score(self):
        pass

