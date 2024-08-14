import pandas as pd
import numpy as np
import copy
import random
import math

class MyBaggingClf:
    def __init__(self, estimator=None, n_estimators=10,
                 max_samples=10, random_state=42):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = list()

    def __str__(self):
        return 'MyBaggingClf class: ' + f'{", ".join(str(i[0])+"="+str(i[1]) for i in self.__dict__.items())}'

    def fit(self, X: pd.DataFrame, y: pd.Series):

        random.seed(self.random_state)
        rows_num_list = list(X.index)
        rows_smpl_cnt = round(X.shape[0]*self.max_samples)
        samples = list()
        for _ in range(self.n_estimators):
            samples.append(random.choices(rows_num_list, k=rows_smpl_cnt))
        for i in range(self.n_estimators):
            X_sample = X.loc[samples[i]]

            y_sample = y.loc[X_sample.index]
            model = copy.deepcopy(self.estimator)
            model.fit(X_sample, y_sample)
            self.estimators.append(model)

    def predict(self,X: pd.DataFrame, type: str):
        if type == 'mean':
            pred_proba = np.array([treeClf.predict_proba(X) for treeClf in self.estimators]).mean(axis=0)
            return np.where(pred_proba > 0.5,1,0)
        elif type == 'vote':
            pred_proba = np.array([treeClf.predict(X) for treeClf in self.estimators]).mean(axis=0)
            return np.where(pred_proba >= 0.5, 1, 0)

    def predict_proba(self, X):
        return np.array([pd.Series(treeClf.predict_proba(X)) for treeClf in self.estimators]).mean(axis=0)



import numpy as np
import pandas as pd
from typing import Optional


class Node:
    def __init__(
        self,
        col: str,
        val: float,
        criterion_val: float = 0,
        elements: int = 0,
        left=None,
        right=None,
    ):
        self.col = col
        self.val = val
        self.left = left
        self.right = right
        self.criterion_val = criterion_val
        self.elements = elements


class Leaf:
    def __init__(
        self,
        name: str = "leaf_left",
        prob_1: float = 0,
        criterion_val: float = 0,
        elements: int = 0,
    ):
        self.name = name
        self.prob_1 = prob_1
        self.criterion_val = criterion_val
        self.elements = elements


class MyTreeClf:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_leafs: int = 20,
        bins: Optional[int] = None,
        criterion: str = "entropy",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max(max_leafs, 2)
        self.bins = bins
        self.criterion = criterion

        self.leafs_cnt = 0
        self.examples_num = 1
        self.required_leafs = 0
        self.tree = {}
        self.delimeters = {}

    def __repr__(self):
        return f"{self.__class__.__name__} class: " + ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items()]
        )

    def _calculate_criterion(self, vals: np.ndarray):
        if self.criterion == "entropy":
            # calculate entropy
            unique_classes, class_counts = np.unique(vals, return_counts=True)
            class_counts = class_counts / len(vals)
            if len(unique_classes) == 1:
                return 0
            criterion_val = -np.sum(class_counts * np.log2(class_counts))
        elif self.criterion == "gini":
            # calculate gini
            _, class_counts = np.unique(vals, return_counts=True)
            class_counts = class_counts / len(vals)
            criterion_val = 1 - np.sum(class_counts**2)

        return criterion_val

    def _calculate_information_gain(
        self, vals: np.ndarray, left: np.ndarray, right: np.ndarray
    ):
        # calculate entropies
        s_0 = self._calculate_criterion(vals[:, 1])
        s_1 = self._calculate_criterion(left[:, 1])
        s_2 = self._calculate_criterion(right[:, 1])

        # calculate inofrmation gane
        ig = s_0 - len(left) / len(vals) * s_1 - len(right) / len(vals) * s_2

        return ig

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_split = (X.columns[0], 0, 0)
        for col_name in X.columns:
            # sort vals
            vals = np.stack((X[col_name].values, y.values), axis=1)
            sorted_indices = np.argsort(vals[:, 0])
            vals = vals[sorted_indices]
            # get delimeters
            delimeters = self.delimeters[col_name]

            for delimeter_value in delimeters:
                left = vals[vals[:, 0] <= delimeter_value]
                right = vals[vals[:, 0] > delimeter_value]
                split_ig = self._calculate_information_gain(vals, left, right)
                if split_ig > best_split[2]:
                    best_split = (col_name, delimeter_value, split_ig)

        return best_split

    def _split(self, X: pd.DataFrame, y: pd.Series, col: str, val: float):
        df = pd.concat((X, y), axis=1)
        # split
        left = df.loc[df[col] <= val]
        right = df.loc[df[col] > val]
        # unpack
        left_X, left_y = left.iloc[:, :-1], left.iloc[:, -1]
        right_X, right_y = right.iloc[:, :-1], right.iloc[:, -1]

        return (left_X, left_y), (right_X, right_y)

    def _add_leaf(self, node: Node, y: pd.Series, direction: str):
        counts = y.value_counts(normalize=True).to_dict()
        criterion_val = self._calculate_criterion(y.values)

        if direction == "left":
            node.left = Leaf("leaf_left", counts.get(1, 0), criterion_val, len(y))
            self.leafs_cnt += 1
            self.required_leafs -= 1
        elif direction == "right":
            node.right = Leaf("leaf_right", counts.get(1, 0), criterion_val, len(y))
            self.leafs_cnt += 1
            self.required_leafs -= 1

    def _check_delimeters_existence(self, X: pd.DataFrame):
        for col_name in X.columns:
            vals = X[col_name].values
            delimeters = self.delimeters[col_name]
            if any([vals.min() <= delimeter <= vals.max() for delimeter in delimeters]):
                return True
        return False

    def _dfs(self, X: pd.DataFrame, y: pd.Series, node: Node, current_depth: int):
        # split by parrent
        (X_left, y_left), (X_right, y_right) = self._split(
            X.copy(), y.copy(), node.col, node.val
        )
        # max depth | min_samples | not delimeters
        if (
            current_depth == self.max_depth
            or len(X) < self.min_samples_split
            or not self._check_delimeters_existence(X)
        ):
            self._add_leaf(node, y_left, "left")
            self._add_leaf(node, y_right, "right")

        # if min_samples_split | _check_delimeters_existence | leafs count
        elif (self.leafs_cnt + self.required_leafs) == self.max_leafs:
            if node.left is None:
                self._add_leaf(node, y_left, "left")
            # right
            if node.right is None:
                self._add_leaf(node, y_right, "right")

        # build new node
        else:
            # find best splits
            left_best_split = self.get_best_split(X_left, y_left)
            right_best_split = self.get_best_split(X_right, y_right)

            # add left split
            if left_best_split[2] == 0 or len(X_left) < self.min_samples_split:
                self._add_leaf(node, y_left, "left")
            else:

                node.left = Node(
                    left_best_split[0],
                    left_best_split[1],
                    self._calculate_criterion(y_left),
                    len(y_left),
                    None,
                    None,
                )
                self.required_leafs += 1
                self._dfs(X_left, y_left, node.left, current_depth + 1)

            # if min_samples_split | _check_delimeters_existence | leafs count
            if (self.leafs_cnt + self.required_leafs) == self.max_leafs:
                if node.left is None:
                    self._add_leaf(node, y_left, "left")
                # right
                if node.right is None:
                    self._add_leaf(node, y_right, "right")
                return None

            # add right split
            if right_best_split[2] == 0 or len(X_right) < self.min_samples_split:
                self._add_leaf(node, y_right, "right")
                return None

            node.right = Node(
                right_best_split[0],
                right_best_split[1],
                self._calculate_criterion(y_right),
                len(y_right),
                None,
                None,
            )
            self.required_leafs += 1
            self._dfs(X_right, y_right, node.right, current_depth + 1)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # find delimeters
        for col_name in X.columns:
            vals = X[col_name].values
            sorted_indices = np.argsort(vals)
            vals = vals[sorted_indices]
            # get delimeters
            _, delimeters = np.unique(vals, return_index=True)
            if self.bins is None or len(delimeters) <= self.bins:
                delimeters = [
                    (vals[delimeter] + vals[delimeter + 1]) / 2
                    for delimeter in delimeters[:-1]
                ]
            else:
                _, delimeters = np.histogram(vals, bins=self.bins)
                delimeters = delimeters[1:-1]

            self.delimeters[col_name] = delimeters

        # get first split
        col, val, _ = self.get_best_split(X.copy(), y.copy())  # (col, val, ig)
        criterion_val = self._calculate_criterion(y.values)
        self.tree = Node(col, val, criterion_val, len(X), None, None)

        # go deeper
        self.required_leafs = 2
        self._dfs(
            X,
            y,
            self.tree,
            1,
        )

    def _tree_traversal(self, d: pd.DataFrame):
        node = self.tree
        while not isinstance(node, Leaf):
            if d[node.col] > node.val:
                node = node.right
            else:
                node = node.left
        return node.prob_1

    def predict(self, X: pd.DataFrame):
        preds = []
        for _, d in X.iterrows():
            prob = self._tree_traversal(d.copy())
            preds.append(prob > 0.5)
        return preds

    def predict_proba(self, X: pd.DataFrame):
        preds = []
        for _, d in X.iterrows():
            prob = self._tree_traversal(d.copy())
            preds.append(prob)
        return preds

    def print_tree(self, node, depth=0):
        if depth == 0:
            print(f"leafs_cnt: {self.leafs_cnt}")
        if isinstance(node, Leaf):
            print("  " * depth, end="")
            print(node.name, "-", node.prob_1)
        elif isinstance(node, Node):
            print("  " * depth, end="")
            print(node.col, ">", node.val)
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)




class MyKNNClf():
    def __init__(self,
                 k: int = 3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform'
                 ):

        self.k = k
        self.metric = metric
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.weight = weight # rank distance

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_size = self.X_train.shape

#---------------------МЕТРИКИ-------------------------------------------
    def _euclidean_dist(self, row: pd.Series) -> pd.Series:
        return (row - self.X_train).pow(2).sum(axis=1).pow(.5)

    def _chebyshev_dist(self, row: pd.Series) -> pd.Series:
        return (row - self.X_train).abs().max(axis=1)

    def _manhattan_dist(self, row: pd.Series) -> pd.Series:
        return (row - self.X_train).abs().sum(axis=1)

    def _cosine_dist(self, row: pd.Series) -> pd.Series:
        return 1 - (row * self.X_train).sum(axis=1) / (np.power(row.pow(2).sum(), 0.5) *
                                                       self.X_train.pow(2).sum(axis=1).pow(0.5))
#--------------------Взешенный KNN------------------------------------------
    def _rank(self, sorted_obj: pd.Series) -> float:
        weight_rank = 1/(np.arange(sorted_obj.shape[0]) + 1)
        Q1 = (weight_rank * y[sorted_obj.index].values).sum()/weight_rank.sum()
        return Q1

    def _distance(self, sorted_obj: pd.Series) -> float:
        weight_rank = 1/sorted_obj.values
        Q1 = (weight_rank * y[sorted_obj.index].values).sum()/weight_rank.sum()
        return Q1

    def _uniform(self, sorted_obj: pd.Series) -> float:
        return y[sorted_obj.index].mean()

#-----------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> int:
        return (X.apply(self._proba_row, axis=1) >= 0.5).astype(int)
#----------------------------------------------------------------
    def _proba_row(self, row: pd.Series) -> float:
        # расчет метрики
        sorted_obj = getattr(self, f'_{self.metric}_dist')(row).sort_values().head(self.k)
        # расчет веса объекта
        return (getattr(self, f'_{self.weight}')(sorted_obj))

    def predict_proba(self, X: pd.DataFrame) -> float:
        return X.apply(self._proba_row, axis=1)



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





