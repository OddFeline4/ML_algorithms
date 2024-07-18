import pandas as pd
import numpy as np

class MyLineReg():
    def __init__(self,weights=None,**kwargs):
        for key in kwargs:
            self.__setattr__(key,kwargs[key])
        self.weights = weights

    def fit(self, X, y, verbose):
        count_verbose = 1
        X.insert(0,'w0',np.full(shape=X.shape[0],fill_value=1))
        self.weights = np.full(shape=X.shape[1],fill_value=1)
        for i in range(self.__dict__['n_iter']):
            y_pred = X.dot(self.weights)
            error = ((y_pred - y)**2).sum()/X.shape[0]
            y_diff = y_pred - y
            gradient = y_diff.dot(X)*(2/X.shape[0])
            self.weights = self.weights - self.__dict__['learning_rate']*gradient
            if count_verbose % verbose == 0:
                print(count_verbose,'|',error)
            count_verbose+=1

    def get_coef(self):
        return self.weights[1:]

    def __str__(self):
        return f'MyLineReg class: {", ".join([str(x[0])+"="+str(x[1]) for x in self.__dict__.items()])}'
