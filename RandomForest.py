import pandas as pd
import numpy as np
import random


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='mse'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1

        self.__sum_tree_values = 0
        self.bins = bins
        self.X_thresholds = None
        self.criterion = criterion
        self.fi = {}

    def get_sum(self):
        return self.__sum_tree_values

    def fit(self, X, y):
        self.fi = {col: 0 for col in X.columns}
        if not(self.bins is None):
            thresholds = {}
            for col in X.columns:
                unique = pd.Series(X[col].unique()).sort_values()
                if len(unique) <= self.bins - 1:
                    thresholds[col] = unique
                else:
                    _, thresholds[col] = np.histogram(X[col],bins=self.bins)
                    thresholds[col] = (thresholds[col])[1:-1]
            self.X_thresholds = pd.DataFrame.from_dict(thresholds)

        self.tree = None

        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, split_value, _ = self.get_best_split(X_root, y_root)
            if depth >= self.max_depth or \
              len(y_root) < self.min_samples_split or \
              (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs) or col_name is None:
                root.side = side
                root.value_leaf = y_root.mean()
                self.__sum_tree_values += root.value_leaf
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            X_left = X_root.loc[X_root[col_name] <= split_value]
            y_left = y_root.loc[X_root[col_name] <= split_value]

            X_right = X_root.loc[X_root[col_name] > split_value]
            y_right = y_root.loc[X_root[col_name] > split_value]
            if self.criterion == 'mse':
                I0 = self.calculate_mse(y_root)
                I_left = self.calculate_mse(y_left)
                I_right = self.calculate_mse(y_right)
            else:
                I0 = self.calculate_mae(y_root)
                I_left = self.calculate_mae(y_left)
                I_right = self.calculate_mae(y_right)
            importance = (len(y_root)/len(y))*(I0-I_left*len(y_left)/len(y_root)-I_right*len(y_right)/len(y_root))
            self.fi[col_name] += importance
            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)

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

    def calculate_mse(self, y: pd.Series):
        y_mean = y.mean()
        return ((y-y_mean)**2).sum()/y.count()

    def calculate_mae(self, y):
        y_median = y.median()
        return ((y-y_median).abs()).sum()/y.count()

    def best_mae_split(self, X, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None

        start_mae = self.calculate_mae(y)
        for col in columns:
            if self.bins is None:
                uniq_values = X[col].sort_values().unique()
                uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
                for uniq_value in uniq:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    mae_left = self.calculate_mae(y_left)
                    mae_right = self.calculate_mae(y_right)
                    info_profit_step = start_mae - (y_left.count()/y.count())*mae_left - (y_right.count()/y.count())*mae_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
            else:
                edges = self.X_thresholds[col]
                for uniq_value in edges:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    if len(indexes_left) == 0 or len(indexes_right) == 0:
                        continue
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    mae_left = self.calculate_mae(y_left)
                    mae_right = self.calculate_mae(y_right)
                    info_profit_step = start_mae - (y_left.count()/y.count())*mae_left - (y_right.count()/y.count())*mae_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
        return col_name, split_value, info_profit



    def get_best_split(self, X, y):
        if self.criterion == 'mse':
            col_name, split_value, info_profit = self.best_mse_split(X,y)
        else:
            col_name, split_value, info_profit = self.best_mae_split(X,y)
        return col_name, split_value, info_profit

    def best_mse_split(self, X, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None
        start_mse = self.calculate_mse(y)
        for col in columns:
            if self.bins is None:
                uniq_values = X[col].sort_values().unique()
                uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
                for uniq_value in uniq:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    mse_left = self.calculate_mse(y_left)
                    mse_right = self.calculate_mse(y_right)
                    info_profit_step = start_mse - (y_left.count()/y.count())*mse_left - (y_right.count()/y.count())*mse_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
            else:
                edges = self.X_thresholds[col]
                for uniq_value in edges:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    if len(indexes_left) == 0 or len(indexes_right) == 0:
                        continue
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    mse_left = self.calculate_mse(y_left)
                    mse_right = self.calculate_mse(y_right)
                    info_profit_step = start_mse - (y_left.count()/y.count())*mse_left - (y_right.count()/y.count())*mse_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
        return col_name, split_value, info_profit

    def __str__(self):
        return 'MyTreeReg class: '+ f'{", ".join(str(i[0])+"="+str(i[1]) for i in self.__dict__.items())}'

    def predict(self,X: pd.DataFrame):
        result = list()
        for idx in X.index:
            object = X.loc[[idx]]  # DataFrame
            tree_node = self.tree
            while tree_node.value_leaf is None:
                if object[tree_node.feature].loc[idx] <= tree_node.value_split:
                    tree_node = tree_node.left
                else:
                    tree_node = tree_node.right
            result.append(tree_node.value_leaf)
        return result


class MyForestReg:
    def __init__(self,n_estimators: int =10, max_features: float = 0.5,
                 max_samples:float = 0.5, random_state: int = 42, max_depth:int = 5,
                 min_samples_split:int = 2, max_leafs: int=20, bins:int = 16,oob_score=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.random_state = random_state
        self.leafs_cnt = 0
        self.forest = {}
        self.fi = {}
        self.oob_score = oob_score
        self.oob_score_ = 0

    def count_metric(self, y_pred, y_true, num):
        if self.oob_score == 'mse':
            return ((y_true - y_pred)**2).sum()/num
        elif self.oob_score == 'mae':
            return abs((y_true - y_pred).abs()).sum()/num
        elif self.oob_score == 'rmse':
            return np.sqrt(((y_true - y_pred)**2).sum()/num)
        elif self.oob_score == 'r2':
            return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
        elif self.oob_score == 'mape':
            return (100/num)*(abs((y_true-y_pred)/y_true)).sum()
        else:
            return None


    def fit(self,X:pd.DataFrame ,y:pd.Series):
        self.fi = {col: 0 for col in X.columns}
        random.seed(self.random_state)
        init_rows =X.shape[0]
        init_cols = list(X.columns)                         # list ?
        rows_cnt = round(init_rows*self.max_samples)
        cols_cnt = round(len(init_cols)*self.max_features)
        pred_frame = pd.Series(np.zeros(X.shape[0])).to_frame('start_col')
        for i in range(self.n_estimators):
            cols_idx = list(random.sample(init_cols,cols_cnt))
            rows_indx = list(random.sample(range(init_rows),rows_cnt))
            y_sample = y[rows_indx]
            X_sample = X[cols_idx].loc[rows_indx]
            oob_samples = X[cols_idx].loc[list(set(X.index) - set(X_sample.index))]
            tree = MyTreeReg(self.max_depth, self.min_samples_split,
                             self.max_leafs, self.bins)
            tree.fit(X_sample,y_sample)
            self.leafs_cnt += tree.leafs_cnt
            self.forest[f'{tree}_{i+1}'] = tree
            for feature in self.fi:
                tree.fi.setdefault(feature, 0)
                self.fi[feature] += tree.fi[feature]*(rows_cnt/init_rows)
            y_oob_pred = pd.Series(tree.predict(oob_samples))
            y_oob_pred.name = f'tree_{i+1}'
            y_oob_pred.index = oob_samples.index
            pred_frame = pred_frame.join(y_oob_pred,how='outer')
        def make_avg(row):
            total = 0
            count = 0
            for col in pred_frame.columns:
                if not str(row[col]).isalpha():
                    total += row[col]
                    count += 1
            return total/count
        pred_frame = pred_frame.drop('start_col',axis='columns')
        y_pred = pd.Series(pred_frame.apply(make_avg,axis=1)).sort_index()
        y_test = y[y_pred.index].sort_index()
        self.oob_score_ = self.count_metric(y_pred,y_test,y_pred.count())



    def predict(self, X):
        pred = np.array([tree.predict(X) for tree in self.forest]).mean(axis=0)
        return pred
    def predict(self,X:pd.DataFrame):
        y = pd.Series(np.zeros(X.shape[0]))
        for key in self.forest:
            curr_tree = self.forest[key]
            y_curr_pred = curr_tree.predict(X)
            y += y_curr_pred
        y /= self.n_estimators
        return y


    def __str__(self):
        return 'MyForestReg class: ' + f'{", ".join([str(i[0])+"="+str(i[1]) for i in self.__dict__.items()])}'



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
# regressor = MyForestReg(oob_score='mae')
# regressor.fit(X_train,y_train)
# print(regressor.oob_score_)
# y_pred = regressor.predict(X_test)
#
# def r2_metric(y_true,y_pred):
#     return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
# print(r2_metric(y_test,y_pred))
# print(regressor.fi)
