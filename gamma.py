from sklearn.neighbors import NearestNeighbors
import numpy as np

class Gamma():
    def __init__(self, gamma = 0.5, nb_nn = 3):
        self.gamma = gamma
        self.nb_nn = nb_nn
    
    def fit(self, X, y):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        self.nn_pos_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_.fit(self.X_[self.y_ == 1])
        
        self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_neg_.fit(self.X_[self.y_ != 1])
        
        return self
        
    def predict(self, X):            
        distance_test_to_positive = self.nn_pos_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]
        
        return  [np.count_nonzero((np.argsort(np.concatenate((distance_test_to_positive[i]*self.gamma,distance_test_to_negative[i])))<self.nb_nn)[:self.nb_nn])>=(self.nb_nn//2+1) for i in range(len(X))]

