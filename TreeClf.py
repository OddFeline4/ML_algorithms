import  pandas as pd
import numpy as np


class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None

class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, edge=0.5, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1

        self.__sum_tree_values = 0
        self.edge = edge
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

            proportion_ones = len(y_root[y_root == 1]) / len(y_root) if len(y_root) else 0

            if proportion_ones == 0 or proportion_ones == 1 or depth >= self.max_depth or \
              len(y_root) < self.min_samples_split or \
              (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs) or col_name is None:
                root.side = side
                root.value_leaf = proportion_ones
                self.__sum_tree_values += root.value_leaf
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            X_left = X_root.loc[X_root[col_name] <= split_value]
            y_left = y_root.loc[X_root[col_name] <= split_value]

            X_right = X_root.loc[X_root[col_name] > split_value]
            y_right = y_root.loc[X_root[col_name] > split_value]
            if self.criterion == 'entropy':
                I0 = self.calculate_entropy(y_root)
                I_left = self.calculate_entropy(y_left)
                I_right = self.calculate_entropy(y_right)
            else:
                I0 = self.calculate_gini(y_root)
                I_left = self.calculate_gini(y_left)
                I_right = self.calculate_gini(y_right)
            importance = (len(y_root)/len(y))*(I0-I_left*len(y_left)/len(y_root)-I_right*len(y_right)/len(y_root))
            self.fi[col_name] += importance
            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)
        # self.print_tree()

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

    def calculate_entropy(self, y):
        if y.sum() == y.count() or y.sum() == 0:
            return 0
        else:
            return -1 *((y.sum()/y.count())*np.log2(y.sum()/y.count()) + ((y.count()-y.sum())/y.count())*np.log2(((y.count()-y.sum())/y.count())))

    def calculate_gini(self, y):
        prob_0 = (y.count()-y.sum())/y.count()
        prob_1 = y.sum()/y.count()
        return 1 - prob_1**2 - prob_0**2

    def best_gini_split(self, X, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None

        start_gini = self.calculate_gini(y)
        for col in columns:
            if self.bins is None:
                uniq_values = X[col].sort_values().unique()
                uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
                for uniq_value in uniq:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    G_left = self.calculate_gini(y_left)
                    G_right = self.calculate_gini(y_right)
                    info_profit_step = start_gini - (y_left.count()/y.count())*G_left - (y_right.count()/y.count())*G_right
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
                    G_left = self.calculate_gini(y_left)
                    G_right = self.calculate_gini(y_right)
                    info_profit_step = start_gini - (y_left.count()/y.count())*G_left - (y_right.count()/y.count())*G_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
        return col_name, split_value, info_profit



    def get_best_split(self, X, y):
        if self.criterion == 'entropy':
            col_name, split_value, info_profit = self.best_entropy_split(X,y)
        else:
            col_name, split_value, info_profit = self.best_gini_split(X,y)
        return col_name, split_value, info_profit

    def best_entropy_split(self, X, y):
        columns = X.columns
        info_profit = 0
        split_value = None
        col_name = None

        start_entropy = self.calculate_entropy(y)
        for col in columns:
            if self.bins is None:
                uniq_values = X[col].sort_values().unique()
                uniq = [(uniq_values[i]+uniq_values[i+1])/2 for i in range(len(uniq_values)-1)]
                for uniq_value in uniq:
                    indexes_left = X[X[col] <= uniq_value].index
                    indexes_right = X[X[col] > uniq_value].index
                    y_left = y[indexes_left]
                    y_right = y[indexes_right]
                    S_left = self.calculate_entropy(y_left)
                    S_right = self.calculate_entropy(y_right)
                    info_profit_step = start_entropy - (y_left.count()/y.count())*S_left - (y_right.count()/y.count())*S_right
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
                    S_left = self.calculate_entropy(y_left)
                    S_right = self.calculate_entropy(y_right)
                    info_profit_step = start_entropy - (y_left.count()/y.count())*S_left - (y_right.count()/y.count())*S_right
                    if info_profit_step > info_profit:
                        info_profit = info_profit_step
                        split_value = uniq_value
                        col_name = str(col)
        return col_name, split_value, info_profit

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def predict_proba(self,X: pd.DataFrame):
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

    def predict(self, X: pd.DataFrame):
        result = pd.Series(self.predict_proba(X))
        return list((result > self.edge).astype('int64'))






df = pd.read_csv('data_banknote_authentication.txt', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']
obj = MyTreeClf(max_depth=15,  min_samples_split=20, max_leafs=30, bins=6, criterion='gini')
obj.fit(X,y)
obj.print_tree()
print(obj.get_sum())
print(obj.fi)
# result1 = pd.Series(obj.predict_proba(X))
# result2 = pd.Series(obj.predict(X))
# res = pd.concat([result1,result2],axis='columns')
# print(res)









