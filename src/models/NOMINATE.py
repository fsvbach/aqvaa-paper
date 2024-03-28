import pandas as pd
import numpy as np

from scipy.spatial import distance
from scipy.optimize import minimize

class NOMINATE:
    def __init__(self, legislators, rollcalls, index=None):
        beta = rollcalls.loc[:,'beta'].mean()
        weights = rollcalls.loc[:,['weight1D', 'weight2D']].mean(axis=0).values
        embedding = legislators.loc[:, ['coord1D', 'coord2D']] * weights

        if index is None:
            index = legislators.index
        self.train_embedding = pd.DataFrame(embedding.values, index=index, columns=['x','y'])
        
        items = rollcalls.apply(self.coordsNOM, axis=1, result_type='expand').reset_index(drop=True)
        items.index = items.index.astype(str)
        items.columns = ['x1','x2','y1','y2']

        self.weights = weights
        self.beta    = beta
        self.items   = items

    def coordsNOM(self, row):
        x1 = row['weight1D'] * (row['midpoint1D'] - row['spread1D']) # YAY
        y1 = row['weight2D'] * (row['midpoint2D'] - row['spread2D']) # YAY
        x2 = row['weight1D'] * (row['midpoint1D'] + row['spread1D']) # NAY
        y2 = row['weight2D'] * (row['midpoint2D'] + row['spread2D']) # NAY
        return x1,x2,y1,y2
    
    def predict(self, params, queries=None):
        if queries is None:
            queries = self.items.index
        U1 = self.beta * np.exp( - distance.cdist(params, self.items.loc[queries, ['x1','y1']]) / 2) 
        U2 = self.beta * np.exp( - distance.cdist(params, self.items.loc[queries, ['x2','y2']]) / 2)

        return np.exp(U1) / (np.exp(U1) + np.exp(U2))

    # take transformed coordinates directly (that's why for border we divide by weights)
    def objective(self, params, answers):
        probs = self.predict(params.reshape(-1, 2), answers.index)
        loss  = np.nansum(-np.log(np.abs( answers.values - 1 + probs)),axis=1)
        loss[np.where(np.linalg.norm(params.reshape(-1, 2)/self.weights, axis=1) > 1)] = 1e4
        return loss
        
    def nan_objective(self, params, answers):
        loss = self.objective(params, answers)
        loss[np.where(loss>=1e4)] = np.nan
        return loss
    
    def nan_predict(self, params, answers):
        predictions = self.predict(params, answers)
        predictions[np.where(np.linalg.norm(params.reshape(-1, 2)/self.weights, axis=1) > 1)] = np.nan
        return predictions

    def encode(self, row):
        answers = row.loc[~row.isna()]
        res = minimize(self.objective, np.zeros(2), args=(answers,), method='BFGS')
        return res.x[0], res.x[1], res.fun