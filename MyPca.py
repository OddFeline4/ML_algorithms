import numpy.ma
import pandas as pd
from numpy import linalg

class MyPCA:
    def __init__(self,n_components=3):
        self.n_components = n_components

    def fit_transform(self,X:pd.DataFrame):
        X_ = X.copy()
        for col in X.columns:
            X_[col] = X_[col] - X_[col].mean()

        var_list = [X_[col].var() for col in X_.columns]
        columns = list(X_.columns)


        matrix = [[0 for i in range(len(X.columns))] for j in range(len(X.columns))]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if i == j:
                    matrix[i][j] = var_list[i]
                else:
                     matrix[i][j] = numpy.ma.cov(X_[columns[i]],X_[columns[j]])[0][1]
        val,vec = numpy.linalg.eigh(matrix)
        result = [(val[i],vec[i]) for i in range(len(val))]
        result.sort(key=lambda x: x[0],reverse=True)
        main_component = list(map(lambda x: x[1], result[:self.n_components]))
        X_result = X_.dot(main_component)
        return X_result


    def __str__(self):
        return  'MyPCA class: ' + f'{", ".join([i[0]+ "=" + str(i[1]) for i in self.__dict__.items()])}'


# ser1 = pd.Series([1,2,9])
# ser2 = pd.Series([6,8,10])
# df = pd.concat([ser1,ser2],axis='columns')
# pca = MyPCA()
# pca.fit_transform(df)
