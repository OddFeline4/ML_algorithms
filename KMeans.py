

import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3,
                 random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.inertia_ = -1
        self.cluster_centers_ = list()
        self.best_cluster = list()
        self.wcss_scores = list()

    def fit(self, X: pd.DataFrame):
        np.random.seed(self.random_state)
        for i in range(self.n_init):
            features_map = {}
            for col in X.columns:

                min_feature = X[col].min()
                max_feature = X[col].max()
                features_map[col] = (min_feature, max_feature)
            start_centroids = [[np.random.uniform(features_map[key][0], features_map[key][1]) for key in features_map] for j in range(self.n_clusters)]

            for step in range(self.max_iter):

                dist_table = self.make_dist_table(X,start_centroids)

                best_distance_list = pd.Series([int(dist_table.loc[j].idxmin()) for j in range(X.shape[0])],name='num_cluster')

                new_centroids = [list() for _ in range(self.n_clusters)]

                for claster in best_distance_list.unique():
                    samples = X.loc[(best_distance_list[best_distance_list == claster]).index]
                    new_centroids[claster] = list(samples.mean(axis=0))
                for ind in range(len(new_centroids)):
                    if len(new_centroids[ind]) == 0:
                        new_centroids[ind] = start_centroids[ind]




                if step != 0 and (pd.Series(new_centroids).sum() == pd.Series(start_centroids).sum()):
                    self.wcss_scores.append(self.make_wcss_score(X,new_centroids,best_distance_list))
                    break
                start_centroids = new_centroids
                if step == self.max_iter - 1:
                    self.wcss_scores.append(self.make_wcss_score(X,new_centroids,best_distance_list))



            self.cluster_centers_.append(start_centroids)
        best_index = self.wcss_scores.index(min(self.wcss_scores))
        self.inertia_ = self.wcss_scores[best_index]
        self.cluster_centers_= self.cluster_centers_[best_index]

    def make_wcss_score(self, X: pd.DataFrame, centroids: list, nums: pd.Series):
        wcss_score = 0
        for idx in range(X.shape[0]):
            wcss_score += ((X.loc[idx] - np.array(centroids[nums[idx]]))**2).sum()
        return wcss_score


    def make_dist_table(self, X, start_centroids):
            for idx in range(self.n_clusters):
                dist_list = pd.Series([np.sqrt(((X.loc[j]-start_centroids[idx])**2).sum()) for j in range(X.shape[0])],name=idx)     #с нуля ли ? строковый?
                if idx == 0:
                        clust_frame_n = dist_list.to_frame()
                else:
                        clust_frame_n[f'{idx}'] = dist_list

            return clust_frame_n

    def predict(self, X:pd.DataFrame):
        dist_table = self.make_dist_table(X,self.cluster_centers_)
        best_distance_list = pd.Series([int(dist_table.loc[j].idxmin())+1 for j in range(X.shape[0])],name='num_cluster')
        return list(best_distance_list)



    def __str__(self):
        return 'MyKMeans class: ' + f'{", ".join([i[0]+ "=" + str(i[1]) for i in self.__dict__.items()])}'





# df = pd.read_csv('data_banknote_authentication.txt', header=None)
# df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
# X, y = df.iloc[:,:4], df['target']
# # X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.25)
# obj = MyKMeans()
# obj.fit(X)
# # print(obj.inertia_)
# # print(obj.cluster_centers_)
# print((round(obj.inertia_,10), np.array(obj.cluster_centers_).sum().round(10)))
