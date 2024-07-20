
import numpy
import pandas
import pandas as pd
import numpy as np
import random

class MyLineReg():
    def __init__(self, weights=None, metric: str =None, reg=None,\
        l1_coef: float = 0, l2_coef: float = 0, sgd_sample=None, random_state: int = 42,\
        **kwargs):
        for key in kwargs:
            self.__setattr__(key,kwargs[key])
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    #
    # Метод расссчёта одной из метрик качества ( не путать с лосс функцией)
    def count_metric(self, y_pred, y_true, type, num):
        if type == 'mse':
            return ((y_true - y_pred)**2).sum()/num
        elif type == 'mae':
            return abs((y_true - y_pred).abs()).sum()/num
        elif type == 'rmse':
            return np.sqrt(((y_true - y_pred)**2).sum()/num)
        elif type == 'r2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
        elif type == 'mape':
            return (100/num)*(abs((y_true-y_pred)/y_true)).sum()

    # Метод ля рассчёта лосс функции ошибки (её минимизируем)
    def calculate_loss(self, y_pred: numpy.array, y_true: numpy.array, size: int):
        if self.reg == 'l1':
            return ((y_true-y_pred)**2).sum()/size + self.l1_coef*(abs(self.weights).sum())
        elif self.reg == 'l2':
            return ((y_true-y_pred)**2).sum()/size + self.l2_coef*((self.weights**2).sum())
        elif self.reg == 'elasticnet':
            return ((y_true-y_pred)**2).sum()/size + self.l2_coef*((self.weights**2).sum()) + self.l1_coef*(abs(self.weights).sum())
        else:
            return ((y_true - y_pred)**2).sum()/size

    # Интод для вычисления градиента в каждой итерации
    def calculate_grad(self, y_diff: numpy.array, x_array: pandas.DataFrame, size: int):
        if self.reg == 'l1':
            return (y_diff.dot(x_array)*2)/size + self.l1_coef*(np.sign(self.weights))
        elif self.reg == 'l2':
            return (y_diff.dot(x_array)*2)/size + self.l2_coef*2*self.weights
        elif self.reg == 'elasticnet':
            return (y_diff.dot(x_array)*2)/size + self.l1_coef*(np.sign(self.weights)) + self.l2_coef*2*self.weights
        else:
            return (y_diff.dot(x_array)*2)/size

    # Метод для подготовки данных к стохастическому градиенту
    def prepare_data_for_grad(self,X,y_diff):
        if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                x_data = X.iloc[sample_rows_idx]
                size = self.sgd_sample
                y_data = y_diff.iloc[sample_rows_idx]
        elif isinstance(self.sgd_sample, float):
                batch_size = round(self.sgd_sample * X.shape[0])
                sample_rows_idx = random.sample(range(X.shape[0]), batch_size)
                x_data = X.iloc[sample_rows_idx]
                size = batch_size
                y_data = y_diff.iloc[sample_rows_idx]
        else:
                size = X.shape[0]
                x_data = X
                y_data = y_diff
        return x_data,y_data,size

    # Метод, запускающий обучение модели
    def fit(self, X, y, verbose):
        random.seed(self.random_state)
        count_verbose = 1
        X.insert(0,'w0',np.full(shape=X.shape[0],fill_value=1))
        self.weights = np.full(shape=X.shape[1],fill_value=1)
        for i in range(self.__dict__['n_iter']):
            y_pred = X.dot(self.weights)
            error = self.calculate_loss(y_pred,y,X.shape[0])
            y_diff = y_pred - y
            x_data, y_data, size = self.prepare_data_for_grad(X,y_diff)
            gradient = self.calculate_grad(y_data,x_data,size)
            if isinstance(self.__dict__['learning_rate'],(int,float)):
                step = self.__dict__['learning_rate']
            else:
                step = self.__dict__['learning_rate'](i+1)

            self.weights = self.weights - step*gradient
            if not self.metric is None:
                result = self.count_metric(X.dot(self.weights),y,self.metric,X.shape[0])
            else:
                result = ''
            if verbose and count_verbose % verbose == 0:
                print(i+1, '|', error, '|', self.metric+':', result,'|','learning_rate',step)
            count_verbose += 1
            self.best_score = result


    # Метод, врзвращающий предсказания модели
    def predict(self,X:pd.DataFrame):
        if 'w0' not in X.columns:
            X.insert(0,'w0',np.full(shape=X.shape[0],fill_value=1))
        y_predict = X.dot(self.weights)
        return y_predict

    def get_best_score(self):
        return self.best_score

    def get_coef(self):
        return self.weights[1:]

    def __str__(self):
        return f'MyLineReg class: {", ".join([str(x[0])+"="+str(x[1]) for x in self.__dict__.items()])}'



#
# from sklearn import datasets
#
# diabets = datasets.load_diabetes()
# X = pd.DataFrame(diabets['data'])
# y = diabets['target']
# regression = MyLineReg(metric='mape',learning_rate=0.13,n_iter=5, reg='l2',l1_coef=0.15,l2_coef=0.15,sgd_sample=0.2)
# regression.fit(X,y,15)
# print(regression.get_best_score())
