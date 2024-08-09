import pandas as pd
import numpy as np
import copy
import random

class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators=10,
                 max_samples=10, random_state=42, oob_score=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = list()
        self.oob_score = oob_score
        self.oob_score_ = 0

    def __str__(self):
        return 'MyBaggingReg class: ' + f'{", ".join(str(i[0])+"="+str(i[1]) for i in self.__dict__.items())}'

    def fit(self, X: pd.DataFrame, y: pd.Series):
        oob_score_dict = {}
        random.seed(self.random_state)
        rows_num_list = list(X.index)
        rows_smpl_cnt = round(X.shape[0]*self.max_samples)
        samples = list()
        for _ in range(self.n_estimators):
            samples.append(random.choices(rows_num_list, k=rows_smpl_cnt))
        for i in range(self.n_estimators):
            X_sample = X.loc[samples[i]]
            oob_indexes = list(set(X.index) - set(X_sample.index))
            y_sample = y.loc[X_sample.index]
            model = copy.deepcopy(self.estimator)
            model.fit(X_sample, y_sample)
            self.estimators.append(model)
            y_pred_oob = pd.Series(model.predict(X.loc[oob_indexes]))
            y_pred_oob.index = oob_indexes
            for ind in oob_indexes:
                oob_score_dict.setdefault(ind,[])
                oob_score_dict[ind].append(y_pred_oob[ind])
        y_pred = pd.Series([sum(oob_score_dict[key])/len(oob_score_dict[key]) for key in oob_score_dict.keys()])
        y_pred.index = list(oob_score_dict.keys())
        y_test = y[y_pred.index]
        self.oob_score_ = self.count_metric(y_pred,y_test)

    def predict(self, X):
        return np.array([model.predict(X) for model in self.estimators]).mean(axis=0)

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






import math
class MyLineReg:
    def __init__(self,
                 n_iter=100,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,
                 l1_coef=0.0,
                 l2_coef=0.0,
                 sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.metrics = {
            'mae':  lambda y, y_pred: np.mean(np.abs(y - y_pred)),
            'mse':  lambda y, y_pred: np.mean((y - y_pred) ** 2),
            'rmse': lambda y, y_pred: np.sqrt(np.mean((y - y_pred) ** 2)),
            'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
            'r2':   lambda y, y_pred: 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        X = X.copy()
        ones = np.ones(n_samples)
        X.insert(0, 'x_0', ones)

        random.seed(self.random_state)

        self.weights = np.ones(n_features + 1)

        for iter in range(1, self.n_iter + 1):
            y_pred = X @ self.weights
            loss = self.__calc_loss(y, y_pred)

            sample_rows_idx = range(n_samples)
            if isinstance(self.sgd_sample, int):
                sample_rows_idx = random.sample(range(n_samples), self.sgd_sample)
            if isinstance(self.sgd_sample, float):
                k = int(n_samples * self.sgd_sample)
                sample_rows_idx = random.sample(range(n_samples), k)

            X_sample = X.iloc[sample_rows_idx]
            y_sample = y.iloc[sample_rows_idx]

            if callable(self.learning_rate):
                lr = self.learning_rate(iter)
            else:
                lr = self.learning_rate

            self.weights -= lr * self.__calc_grad(X_sample, y_sample)

            if self.metric is not None:
                self.best_score = self.metrics[self.metric](y_sample, y_pred)

            if verbose and iter % verbose == 0:
                print(f"{iter if iter != 0 else 'start'} | loss: {loss}", f"| {self.metric}: {self.best_score}" if self.metric else '', f"| learning_rate: {lr}")

    def predict(self, X):
        X = X.copy()
        ones = np.ones(X.shape[0])
        X.insert(0, 'x_0', ones)
        return X @ self.weights

    def __calc_loss(self, y, y_pred):
        n_samples, _ = X.shape
        loss = np.sum((y - y_pred) ** 2) / n_samples

        if self.reg == 'l1' or self.reg == 'elasticnet':
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == 'l2' or self.reg == 'elasticnet':
            loss += self.l2_coef * np.sum(self.weights ** 2)

        return loss

    def __calc_grad(self, X, y):
        n_samples, _ = X.shape
        grad = 2 / n_samples * (X.T @ (X @ self.weights - y))

        if self.reg:
            if self.reg == 'l1' or self.reg == 'elasticnet':
                grad += self.l1_coef * np.sign(self.weights)
            if self.reg == 'l2' or self.reg == 'elasticnet':
                grad += self.l2_coef * 2 * self.weights

        return grad

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.best_score

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"




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



class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        self.train_size = None
        self.x_data = None
        self.y_data = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNReg class: {", ".join(item[0]+"="+str(item[1]) for item in self.__dict__.items())}'

    def fit(self, x, y):
        self.x_data = x
        self.y_data = y
        self.train_size = x.shape

    def make_predict_by_weight(self,y_data: pd.Series, distance: pd.Series):
        y_data.reset_index(drop=True,inplace=True)
        if self.weight == 'uniform':
            return y_data.mean()
        elif self.weight == 'rank':
            coeff = sum([1/(i+1) for i in y_data.index])
            class_1_weight = sum([x[0]*(1/(x[1]+1)) for x in zip(y_data,y_data.index)])
        else:
            distance_classes = zip(y_data, distance)
            coeff = sum([1/(i) for i in distance])
            class_1_weight = sum([x[0]*(1/x[1]) for x in distance_classes])
        result = class_1_weight/coeff
        return result

    def find_distance(self,x_: pd.DataFrame, what: str = None):
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
                class_value = self.make_predict_by_weight(self.y_data.iloc[k_closet_index],pd.Series(diff_array).sort_values().head(self.k))
                result.append(class_value)
            return pd.Series(result)

    def predict(self,x_test: pd.DataFrame):
        result = pd.Series(self.find_distance(x_test))
        result.index = x_test.index
        return result



