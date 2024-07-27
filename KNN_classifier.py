import pandas
import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_size = None
        self.x_data = None
        self.y_data = None

    def __str__(self):
        return f'MyKNNClf class: {", ".join(item[0]+"="+str(item[1]) for item in self.__dict__.items())}'

    def fit(self, x, y):
        self.x_data = x
        self.y_data = y
        self.train_size = x.shape

    def predict(self,x_test: pandas.DataFrame):
        result = list()
        for i in range(x_test.shape[0]):
            new_object = x_test.iloc[i]
            diff_array = list()
            for j in range(self.x_data.shape[0]):
                difference = new_object.sub(self.x_data.iloc[j])
                diff_array.append((difference**2).sum()**0.5)
            k_closet_index = pd.Series(diff_array).sort_values().head(self.k).index
            class_value = (self.y_data.iloc[k_closet_index]).mode().max()
            result.append(class_value)
        return pd.Series(result)

    def predict_proba(self,x_test: pandas.DataFrame):
        result = list()
        for i in range(x_test.shape[0]):
            new_object = x_test.iloc[i]
            diff_array = list()
            for j in range(self.x_data.shape[0]):
                difference = new_object.sub(self.x_data.iloc[j])
                diff_array.append((difference**2).sum()**0.5)
            k_closet_index = pd.Series(diff_array).sort_values().head(self.k).index
            class_value = (self.y_data.iloc[k_closet_index])
            class_value = class_value.sum()/class_value.count()
            result.append(class_value)
        return pd.Series(result)



my_class = MyKNNClf(k=4)
print(my_class)


