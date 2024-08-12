import numpy as np
import pandas as pd
import random

class MyBoostReg:
    def __init__(self, n_estimators: int = 10, learning_rate = 0.1,
                 max_depth=5,min_samples_split=2,max_leafs=20,bins=16,loss='MSE',metric = None,
                 max_samples=0.5, max_features=0.5, random_state= 42,reg=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.pred_0 = 0
        self.trees = list()
        self.loss = loss
        self.metric = metric
        self.best_score = 0
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.reg = reg
        self.leaf_cnt = 0
        self.fi ={}

    def count_metric(self,y_true ,y_pred ,type):
        if type == 'MSE':
            return ((y_true - y_pred)**2).sum()/len(y_true)
        elif type == 'MAE':
            return abs((y_true - y_pred).abs()).sum()/len(y_true)
        elif type == 'RMSE':
            return np.sqrt(((y_true - y_pred)**2).sum()/len(y_true))
        elif type == 'R2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
        elif type == 'MAPE':
            return (100/len(y_true))*(abs((y_true-y_pred)/y_true)).sum()
        else:
            return None

    def __str__(self):
        return 'MyBoostReg class: '+ f'{", ".join([str(i[0])+"="+str(i[1]) for i in self.__dict__.items()])}'


    def fit(self, X: pd.DataFrame, y:pd.Series, X_eval = None, y_eval=None, early_stopping=None, verbose=None):
        best_scores = list()
        scores = list()
        stopping_step = 0
        self.fi = {col:0 for col in X.columns}
        random.seed(self.random_state)
        if self.loss == 'MSE':
            self.pred_0 = y.mean()  # среднее по таргету
        elif self.loss == 'MAE':
            self.pred_0 = y.median()
        y_pred_now = self.pred_0
        for i in range(self.n_estimators):
                init_cols = list(X.columns)
                init_rows = range(X.shape[0])
                cols_sample_cnt = round(self.max_features*len(X.columns))
                rows_sample_cnt = round(self.max_samples*X.shape[0])
                col_idx = random.sample(init_cols,cols_sample_cnt)
                row_idx = random.sample(init_rows,rows_sample_cnt)
                tree = MyTreeReg(max_depth=self.max_depth,max_leafs=self.max_leafs,
                             min_samples_split=self.min_samples_split,bins=self.bins)
                y_for_change = y - y_pred_now
                if self.loss == 'MSE':
                    y_antigrad = (-2)*(y-y_pred_now)
                else:
                    y_antigrad = np.sign(y-y_pred_now)
                X_train = X[col_idx].loc[row_idx]
                y_antigrad_train = y_antigrad[row_idx]
                y_for_change_train = y_for_change[row_idx]
                tree.fit_for_bosting(X_train, y_antigrad_train, y_for_change_train, type=self.loss,y_size=len(y),leafs=self.leaf_cnt,reg=self.reg)
                for key in tree.fi:
                    self.fi[key] += tree.fi[key]
                self.leaf_cnt += tree.leafs_cnt

                new_pred = pd.Series(tree.predict(X))
                if isinstance(self.learning_rate,(int,float)):
                    y_pred_now += self.learning_rate*new_pred
                else:
                    y_pred_now += self.learning_rate(i+1)*new_pred
                self.trees.append(tree)

                if early_stopping:
                    test_score = self.count_metric(y_eval,self.predict(X_eval),type= self.metric if self.metric else self.loss)
                    scores.append(test_score)
                    if len(scores) > 1:
                        if (self.metric and self.metric != 'R2') or not self.metric:
                            if scores[-1] >= scores[-2]:
                                stopping_step += 1
                            else:
                                stopping_step = 0
                        else:
                            if scores[-1] <= scores[-2]:
                                stopping_step += 1
                            else:
                                stopping_step = 0
                if stopping_step == early_stopping:
                    self.trees = self.trees[:-stopping_step]
                    best_scores = best_scores[:-stopping_step]
                    break

                if verbose and i % verbose == 0:
                    print(f'{i}. Loss[{self.loss}]: {self.count_metric(y,y_pred_now,type=self.loss)} |'
                          f' {self.metric}: {self.count_metric(y,y_pred_now,type=self.metric)}|'
                          f' EvalMetric[{self.metric if self.metric else self.loss}]:  {test_score}')
                best_scores.append(self.count_metric(y,y_pred_now,type=self.metric if self.metric else self.loss))

        self.best_score = best_scores[-1]



    def predict(self, X: pd.DataFrame):
        if isinstance(self.learning_rate,(int,float)):
            y_preds = self.learning_rate*np.array([tree.predict(X) for tree in self.trees]).sum(axis=0) + self.pred_0
        else:
            y_preds = np.array([self.trees[i].predict(X)*self.learning_rate(i+1) for i in range(self.n_estimators)]).sum(axis=0) + self.pred_0
        return y_preds



class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.fi = {}

    def fit_for_bosting(self, X, y, y_new_value, type, leafs, reg, y_size=None):
        if y_size is None:
            y_size = len(y)
        self.tree = None
        self.split_values = {}
        self.fi = {col: 0 for col in X.columns}

        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()

            y_root_unique_size = len(y_root.unique())
            if y_root_unique_size == 0 or y_root_unique_size == 1 or \
              depth >= self.max_depth or len(y_root) < self.min_samples_split \
              or (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                if type == 'MSE':
                    root.value_leaf = y_new_value[y_root.index].mean()+leafs*reg
                else:
                    root.value_leaf = y_new_value[y_root.index].median()+leafs*reg
                return root

            col_name, split_value, gain = self.get_best_split(X_root, y_root)

            self.fi[col_name] += gain * len(y_root) / y_size

            X_left = X_root[X_root[col_name] <= split_value]
            y_left = y_root[X_root[col_name] <= split_value]

            X_right = X_root[X_root[col_name] > split_value]
            y_right = y_root[X_root[col_name] > split_value]

            if len(X_left) == 0 or len(X_right) == 0:
                root.side = side
                root.value_leaf = y_root.mean()
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)


        self.tree = create_tree(self.tree, X, y)

    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            node = self.tree
            while node.feature:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.value_leaf)
        return np.array(y_pred)

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value_leaf}")

    def get_best_split(self, X, y, y_size=None):
        if y_size is None:
            y_size = len(y)

        mse_0 = self.mse(y)

        col_name = None
        split_value = None
        gain = -float('inf')

        for col in X.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(X[col])
                if self.bins is None or len(x_unique_values) - 1 < self.bins:
                    self.split_values[col] = np.array([(x_unique_values[i - 1] + \
                    x_unique_values[i]) / 2 for i in range(1, len(x_unique_values))])
                else:
                    _, self.split_values[col] = np.histogram(X[col], bins=self.bins)
                    self.split_values[col] = self.split_values[col][1:-1]

            for split_value_i in self.split_values[col]:
                mask = X[col] <= split_value_i
                left_split, right_split = y[mask], y[~mask]

                mse_left = self.mse(left_split)
                mse_right = self.mse(right_split)

                weight_left = len(left_split) / y_size
                weight_right = len(right_split) / y_size

                mse_i = weight_left * mse_left + weight_right * mse_right

                gain_i = mse_0 - mse_i
                if gain < gain_i:
                    col_name = col
                    split_value = split_value_i
                    gain = gain_i

        return col_name, split_value, gain

    def mse(self, t):
        t_mean = np.mean(t)
        return ((t - t_mean) ** 2).mean()

    def __str__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"



# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# #
# data = load_diabetes(as_frame=True)
# X, y = data['data'], data['target']
# X = X.drop('sex',axis='columns')
# X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
# # regressor = MyForestReg(n_estimators=30)
# X_train.reset_index(inplace=True,drop=True)
# X_test.reset_index(inplace=True,drop=True)
# y_train = y_train.reset_index(drop=True)
# y_test = y_test.reset_index(drop=True)
# regressor = MyBoostReg(n_estimators=30,learning_rate=0.05,loss='MSE',metric='R2',max_depth=5,max_leafs=16,max_features=0.6)
# regressor.fit(X_train,y_train,verbose=2,X_eval=X_test,y_eval=y_test,early_stopping=2)
# y_pred = regressor.predict(X_test)
#
# def r2_metric(y_true,y_pred):
#     return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
# print(r2_metric(y_test,y_pred))
# print(y_test)
