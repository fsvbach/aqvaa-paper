from sklearn.svm import SVC
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def train_LR_models(X, reactions):
    models = {}
    for task in reactions.columns:
        model = LogisticRegression(random_state=0)
        mask = ~reactions.loc[:, task].isna()
        train_data   = X[mask]
        train_labels = round(reactions.loc[mask, task])
        if len(train_labels.unique()) == 2:
            model.fit(train_data, train_labels)
        else: 
            model = None
            print(f"No model fitted for Feature {task}.")
        models[task] = model
    return models


def predict_from_models(models, params, queries=None):
    if queries is None:
        queries = models.keys()

    predictions = []
    for task in queries:
        pred_task = [np.nan]*int(params.size/2)
        if models[task] is not None:
            pred_task = models[task].predict_proba(params.reshape(-1,2))[:,1]
        predictions.append(pred_task)
    
    return np.array(predictions).T