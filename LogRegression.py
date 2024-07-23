
import math
import numpy
import pandas
import pandas as pd
import numpy as np
import random

import sklearn.model_selection


class MyLogReg():
    def __init__(self, n_iter: int = 10, learning_rate = 0.1, weights: np.array = None, metric: str = None,
                 l1_coef: float = 0, l2_coef: float = 0, reg: str = None,sgd_sample=None,random_state = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.eps = 10**(-15)
        self.metric = metric
        self.best_score = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'MyLogReg class: {", ".join([str(x[0])+"="+str(x[1]) for x in self.__dict__.items()])}'

    def loss_function(self,y_true: pd.Series, y_pred: pd.Series, size):
        value_1 =((y_pred+self.eps).apply(lambda z: math.log(z,10)))
        value_2 =((1-y_pred+self.eps).apply(lambda z: math.log(z,10)))
        result =  (-1/size)*(value_1*y_true + value_2*(1-y_true)).sum()
        if self.reg == 'l1':
            return result + self.l1_coef*(np.sign(self.weights)).sum()
        elif self.reg == 'l2':
            return result + 2*self.l2_coef*self.weights.sum()
        elif self.reg == 'elasticnet':
            return result + 2*self.l2_coef*self.weights.sum() + self.l1_coef*(np.sign(self.weights)).sum()
        return result

    def gradient_function(self,y,y_pred,x):
        if self.reg == 'l1':
            return ((y_pred - y).dot(x))/x.shape[0] + self.l1_coef*(np.sign(self.weights))
        elif self.reg == 'l2':
            return ((y_pred - y).dot(x))/x.shape[0] + 2*self.l2_coef*self.weights
        elif self.reg == 'elasticnet':
            return ((y_pred - y).dot(x))/x.shape[0] + 2*self.l2_coef*self.weights + self.l1_coef*(np.sign(self.weights))
        return ((y_pred - y).dot(x))/x.shape[0]

    def prepare_data_for_grad(self,X,y,y_pred):
        if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                x_data = X.iloc[sample_rows_idx]
                size = self.sgd_sample
                y_pred = y_pred.iloc[sample_rows_idx]
                y = y.iloc[sample_rows_idx]
        elif isinstance(self.sgd_sample, float):
                batch_size = round(self.sgd_sample * X.shape[0])
                sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
                x_data = X.iloc[sample_rows_idx]
                size = batch_size
                y_pred = y_pred.iloc[sample_rows_idx]
                y = y.iloc[sample_rows_idx]
        else:
                size = X.shape[0]
                x_data = X
        return x_data,y,y_pred,size

    def fit(self, x: pd.DataFrame, y: np.array, verbose: int = False):
        x.insert(0,'w_0',np.full(x.shape[0], 1))
        self.weights = np.full(x.shape[1],1)
        verbose_count = 1
        for i in range(self.n_iter):
            y_pred = 1/(1+(-1*(x.dot(self.weights))).apply(lambda z: math.exp(z)))
            x_new, y_new, y_pred_new, size_new = self.prepare_data_for_grad(x,y,y_pred)
            loss = self.loss_function(y_new,y_pred_new,x_new.shape[0])
            grad = self.gradient_function(y_new,y_pred_new,x_new)
            if isinstance(self.learning_rate,(int,float)):
                self.weights = self.weights - grad*self.learning_rate
            else:
                self.weights = self.weights - grad*self.learning_rate(i+1)
            y_pred = 1/(1+(-1*(x.dot(self.weights))).apply(lambda z: math.exp(z)))
            metric = None
            if self.metric:
                metric = self.calculate_metric(y_pred,y)
            if verbose and verbose_count % verbose == 0:
                print(i+1, '| loss:', loss,'|', self.metric,metric)
            verbose_count += 1
            self.best_score = metric

    def calculate_metric(self,y_pred,y_true):
        y_prob = y_pred
        y_pred = np.where(y_pred > 0.5, 1, 0)
        if self.metric == 'accuracy':
            return (y_true == y_pred).sum()/len(y_true)
        elif self.metric == 'precision':
            TP = len(list(filter(lambda x: x[0] + x[1] == 2,zip(y_true,y_pred))))
            FP =  len(list(filter(lambda x: x[0] == 0 and x[0] + x[1] == 1,zip(y_true,y_pred))))
            return TP/(TP+FP)
        elif self.metric == 'recall':
            TP = len(list(filter(lambda x: x[0] + x[1] == 2,zip(y_true,y_pred))))
            FN =  len(list(filter(lambda x: x[0] == 1 and x[0] + x[1] == 1,zip(y_true,y_pred))))
            return TP/(TP+FN)
        elif self.metric == 'f1':
            TP = len(list(filter(lambda x: x[0] + x[1] == 2,zip(y_true,y_pred))))
            FP =  len(list(filter(lambda x: x[0] == 0 and x[0] + x[1] == 1, zip(y_true,y_pred))))
            FN =  len(list(filter(lambda x: x[0] == 1 and x[0] + x[1] == 1, zip(y_true,y_pred))))
            precicion = TP/(TP+FP)
            recall = TP/(TP+FN)
            return (2*precicion*recall)/(recall+precicion)
        elif self.metric == 'roc_auc':
            roc_auc_line = sorted(zip(round(y_prob,10),y_true), key=lambda z: (-z[0],-z[1]))
            total_score = 0
            store = 0
            for el in roc_auc_line:
                if el[1] == 1:
                    store += 1
                else:
                    total_score += store
            positive = (y_true == 1).sum()
            negative = (y_true == 0).sum()
            return total_score/(positive*negative)


    def area_lst(self,data):
        result = list()
        for i in range(len(data)):
            flag = False
            for j in range(i):
                if data[i][0] == data[j][0] and data[j][1] == 1:
                    flag = True
            if flag and data[j][1] == 0:
                result.append(data[j][0])
        return result



    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self,x_: pd.DataFrame):
        x = x_.copy()
        x.insert(0,'w_0',np.full(x.shape[0], 1))
        y_pred = 1/(1+(-1*(x.dot(self.weights))).apply(lambda z: math.exp(z)))
        return y_pred

    def predict(self, x_: pd.DataFrame):
        x = x_.copy()
        x.insert(0,'w_0',np.full(x.shape[0], 1))
        y_pred = 1/(1+(-1*(x.dot(self.weights))).apply(lambda z: math.exp(z)))
        classes = np.where(y_pred > 0.5, 1, 0)
        return classes

    def get_best_score(self):
        return self.best_score



from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
x = pd.DataFrame(data['data'])
y = pd.Series(data['target'])
for col in x.columns:
    x[col] = x[col]/x[col].max()
X_train, X_test, y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.25)

classification = MyLogReg(n_iter=500, metric='roc_auc',reg='elasticnet',l1_coef=0.01,l2_coef=0.01,sgd_sample=0.1)
classification.fit(X_train,y_train,2)
print(((classification.predict(X_test)==y_test).sum()/len(y_test)))



