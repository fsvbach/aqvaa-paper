from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .metrics import NearestCandidates, overlap


def make_hashable(series):
    sorted_series = series.sort_index()
    return tuple(sorted_series.index), tuple(sorted_series.values)


class CacheDecorator:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.hit_count = {}
    
    def __call__(self, *args, **kwargs):
        cache_key = make_hashable(*args)
        if cache_key not in self.cache:
            result = self.func(*args, **kwargs)
            self.cache[cache_key] = result
            self.hit_count[cache_key] = 0
        else:
            result = self.cache[cache_key].copy()
            self.hit_count[cache_key] += 1
        
        return result
    
    def get_cache_data(self):
        """Return cache and hit count data for export."""
        return pd.DataFrame([(key, self.hit_count[key], self.cache[key]) for key in self.cache], 
                              columns=['Cache Key', 'Hits', 'Output'])


class SelectionMethod(ABC):
    def __init__(self, use_cache=False, model=None, name='SelectionMethod'):
        self.model = model
        self.name = name
        self.use_cache = use_cache
        if self.use_cache:
            self.compute_weights = CacheDecorator(self.compute_weights)
        
    def __str__(self):
        return self.name

    def evaluate(self, answers, **kwargs):
        assert isinstance(answers, pd.Series), "Answers must be a pandas Series"
        # Compute weights based on the data and fixed_order
        self.user = answers
        self.open_index  = answers.loc[answers.isna()].index
        given_answers = answers.loc[~answers.isna()]
        weights = self.compute_weights(given_answers, **kwargs)
        return pd.Series(weights, index=self.open_index, name=answers.name)

    @abstractmethod
    def compute_weights(self, given_answers, **kwargs):
        pass

class FixedOrder(SelectionMethod):
    def __init__(self, fixed_order, use_cache=True, model=None, name='FixedOrder'):
        super().__init__(use_cache=use_cache, model=model, name=name)
        assert isinstance(fixed_order, list), "fixed_order must be a list"
        self.fixed_order = dict(zip(fixed_order, range(len(fixed_order), 0, -1)))

    def compute_weights(self, given_answers, **kwargs):
        return self.open_index.map(self.fixed_order).values


class ActiveLearner:
    def __init__(self, model, method, train_reactions, test_reactions, reactions=None, k_neighbors=32):
        self.model  = model
        self.method = method
        self.truth     = test_reactions
        self.options = pd.DataFrame([], columns=self.truth.columns, index=self.truth.index, dtype=np.float64)

        if reactions is None:
            reactions = self.options
        self.reactions = reactions.copy()
        self.reactions.apply(self.update_row, axis=1)
        self.predictions = self.reactions.apply(self.model.predict_user, axis=1) 

        self.k_neighbors = k_neighbors
        self.candidates = NearestCandidates(train_reactions, k=k_neighbors)
        true_neighbors = self.truth.apply(self.candidates.recommend, axis=1)
        base_neighbors = self.reactions.apply(self.candidates.recommend, axis=1)
        pred_neighbors = self.predictions.apply(self.candidates.recommend, axis=1)
        self.kNNs = pd.concat([true_neighbors,base_neighbors,pred_neighbors], axis=1)
        self.kNNs.columns = ['True-kNN','Base-kNN','Pred-kNN']
        self.kNNs['Base'] =  self.kNNs.apply(lambda row: overlap(row['True-kNN'], row['Base-kNN']), axis=1)
        self.kNNs['Pred'] =  self.kNNs.apply(lambda row: overlap(row['True-kNN'], row['Pred-kNN']), axis=1)

        self.evaluation  = pd.DataFrame([], columns=['User', 'Question', 'Value', 'Counter', 'Pred-kNN', 'User Pred-kNN', 'Base-kNN', 'User Base-kNN', 'Accuracy', 'Expected Accuracy', 'User Accuracy', 'RMSE', 'Expected RMSE', 'User RMSE', 'Timestamp'])

    def update_row(self, row):
        ### SEND THE WHOLE ROW TO SELECTOR
        result = self.method.evaluate(row)
        ### UPDATES THE WHOLE ROW SO QUESTIONS MISSING IN RESULT BECOME NAN
        self.options.loc[row.name] = result

    def select_question(self):
        return self.options.stack().idxmax()

    def run(self, iterations=None, verbose=1000):
        if iterations is None:
            iterations = (~self.options.isna()).sum().sum()
            
        for i in range(iterations):
            cells_given  = (~self.reactions.isna()).sum().sum()
            ### First, evaluate Accuracy and RMSE and kNNs
            preds = self.predictions[self.reactions.isna()]
            acc = 1 - np.mean(np.abs((np.round(self.truth) - np.round(self.predictions))[self.reactions.isna()]), axis=1)
            acc_exp = preds.map(lambda x: max(x,1-x)).mean().mean()
            rmse = np.sqrt(np.mean(np.square((self.truth - self.predictions))[self.reactions.isna()], axis=1))
            rmse_exp = (preds * (1-preds) * 2).mean().mean()

            ### Then, select the next question
            idx, col = self.select_question()
            value = self.options.loc[idx, col]

            ### Update in reactions dataframe
            self.reactions.loc[idx, col] = self.truth.loc[idx,col]

            ### predict this user again and update predictions
            self.predictions.loc[idx] = self.model.predict_user(self.reactions.loc[idx])

            ### update the entire row
            self.update_row(self.reactions.loc[idx])

            ### Add everything to self.evaluation
            self.evaluation.loc[cells_given] = pd.Series({'User': idx, 'Question': col, 'Pred-kNN': self.kNNs.loc[:,'Pred'].mean(), 'User Pred-kNN': self.kNNs.loc[idx,'Pred'], 'Base-kNN': self.kNNs.loc[:,'Base'].mean(), 'User Base-kNN': self.kNNs.loc[idx,'Base'], 'Value': value, 'Accuracy': np.mean(acc), 'RMSE': np.mean(rmse), 'Expected Accuracy': acc_exp, 'Expected RMSE': rmse_exp, 'User Accuracy': acc.loc[idx], 'User RMSE': rmse.loc[idx], 'Timestamp': pd.Timestamp.now()})           
            
            ### Recompute Nearest Candidates 
            self.kNNs.at[idx, 'Base-kNN'] = self.candidates.recommend(self.reactions.loc[idx])
            self.kNNs.at[idx, 'Pred-kNN'] = self.candidates.recommend(self.predictions.loc[idx])
            self.kNNs.loc[idx, 'Base'] = overlap(self.kNNs.loc[idx, 'Base-kNN'], self.kNNs.loc[idx, 'True-kNN'])
            self.kNNs.loc[idx, 'Pred'] = overlap(self.kNNs.loc[idx, 'Pred-kNN'], self.kNNs.loc[idx, 'True-kNN'])

            if verbose:
                if i % verbose == 0:
                    print(f"Iteration {cells_given}: {idx}-{col} gives {round(np.mean(acc),2)}%")

    def save(self, folder_name, data_name, suffix='server'):
        self.evaluation.to_csv(f'../../results/ALVAA/{folder_name}_{data_name}_{self.model}_{self.method}_{suffix}.csv')