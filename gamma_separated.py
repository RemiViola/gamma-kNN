from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

import numpy as np

class GammaSeparated():
    def __init__(self, gamma_real = 0.5, gamma_synth = 0.5, nb_nn = 3):
        self.gamma_real = gamma_real
        self.gamma_synth = gamma_synth
        self.nb_nn = nb_nn
    
    def fit(self, X, y, OS = SMOTE()):
        self.dim_ = len(X[0])
        self.X_ = X
        self.y_ = y
        
        self.OS_ = OS
        X_os,y_os = self.OS_.fit_resample(self.X_,self.y_)
        
        # Real/Synth Split
        
        index_synth = list(range(len(y_os)))
        for i in range(len(self.y_)):
            index_synth = [i_ for i_ in index_synth if np.any(X_os[i_] != self.X_[i])]
            
        index_real = list(set(list(range(len(y_os))))-set(index_synth))
        
        X_synth, y_synth = X_os[index_synth], y_os[index_synth]
        X_real, y_real = X_os[index_real], y_os[index_real]
        
        # Gamma k-NN
        
        self.nn_pos_real_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_real_.fit(X_real[y_real == 1])
                        
        self.nn_pos_synth_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_pos_synth_.fit(X_synth[y_synth == 1])
        
        self.nn_neg_ = NearestNeighbors(n_neighbors = self.nb_nn)
        self.nn_neg_.fit(X_os[y_os != 1])
        
        return self
    
    def predict(self, X):        
        distance_test_to_real_positive = self.nn_pos_real_.kneighbors(X, return_distance = True)[0]
        distance_test_to_synth_positive = self.nn_pos_synth_.kneighbors(X, return_distance = True)[0]
        distance_test_to_negative = self.nn_neg_.kneighbors(X, return_distance = True)[0]
        
        return [np.count_nonzero((np.argsort(np.concatenate((distance_test_to_real_positive[i]*self.gamma_real,distance_test_to_synth_positive[i]*self.gamma_synth,distance_test_to_negative[i])))<self.nb_nn*2)[:self.nb_nn])>=(self.nb_nn//2+1) for i in range(len(X))]
        
