import pandas
import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = None
        self.x_data = None
        self.y_data = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNClf class: {", ".join(item[0]+"="+str(item[1]) for item in self.__dict__.items())}'

    def fit(self, x, y):
        self.x_data = x
        self.y_data = y
        self.train_size = x.shape

    def make_predict_by_weight(self,y_data: pd.Series, distance: pd.Series):
        y_data.reset_index(drop=True,inplace=True)
        if self.weight == 'uniform':
            return y_data.mode().max()
        elif self.weight == 'rank':
            coeff = sum([1/(i+1) for i in y_data.index])
            class_1_weight = sum([x[0]*(1/(x[1]+1)) for x in zip(y_data,y_data.index)])
        else:
            distance_classes = zip(y_data, distance)
            coeff = sum([1/(i) for i in distance])
            class_1_weight = sum([x[0]*(1/x[1]) for x in distance_classes])
        result = class_1_weight/coeff
        return result > 0.5

    def make_predict_proba_by_weight(self,y_data: pd.Series, distance: pd.Series):
        y_data.reset_index(drop=True,inplace=True)
        if self.weight == 'uniform':
            return y_data.sum()/y_data.count()
        elif self.weight == 'rank':
            coeff = sum([1/(i+1) for i in y_data.index])
            class_1_weight = sum([x[0]*(1/(x[1]+1)) for x in zip(y_data,y_data.index)])
        else:
            distance_classes = zip(y_data, distance)
            coeff = sum([1/(i) for i in distance])
            class_1_weight = sum([x[0]*(1/x[1]) for x in distance_classes])
        result = class_1_weight/coeff
        return result



    def find_distance(self,x_: pandas.DataFrame, what: str = None):

            result = list()
            for i in range(x_.shape[0]):
                new_object = x_.iloc[i]
                diff_array = list()
                for j in range(self.x_data.shape[0]):
                    if self.metric == 'euclidean':
                        difference = new_object.sub(self.x_data.iloc[j])
                        diff_array.append((difference**2).sum()**0.5)
                    elif self.metric == 'manhattan':
                        difference = new_object.sub(self.x_data.iloc[j])
                        diff_array.append(abs(difference).sum())
                    elif self.metric == 'chebyshev':
                        difference = new_object.sub(self.x_data.iloc[j])
                        diff_array.append(abs(difference).max())
                    else:
                        object_ = self.x_data.iloc[j]
                        multi = (new_object * object_).sum()
                        coeff_1 = ((new_object**2).sum())**0.5
                        coeff_2 = ((object_**2).sum())**0.5
                        diff_array.append(1 - multi/(coeff_2*coeff_1))
                k_closet_index = pd.Series(diff_array).sort_values().head(self.k).index
                if what == 'class':
                    class_value = self.make_predict_by_weight(self.y_data.iloc[k_closet_index],pd.Series(diff_array).sort_values().head(self.k))
                else:
                    class_value = self.make_predict_proba_by_weight(self.y_data.iloc[k_closet_index],pd.Series(diff_array).sort_values().head(self.k))
                result.append(class_value)
            return pd.Series(result)

    def predict(self,x_test: pandas.DataFrame):
        result = pd.Series(self.find_distance(x_test,what='class'))
        result.index = x_test.index
        return result

    def predict_proba(self, x_test: pandas.DataFrame):
        result = pd.Series(self.find_distance(x_test,what='proba'))
        result.index = x_test.index
        return result



# my_class = MyKNNClf(k=4)
# print(my_class)
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# data = load_breast_cancer()
# x = pd.DataFrame(data['data'])
# y = pd.Series(data['target'])
# for col in x.columns:
#     x[col] = x[col]/x[col].max()
# X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.25)
#
# classification = MyKNNClf(5,'cousine',weight='rank')
# classification.fit(X_train,y_train)
# print(((classification.predict(X_test))==y_test).sum()/len(y_test))#==y_test).sum()/len(y_test)))
#

