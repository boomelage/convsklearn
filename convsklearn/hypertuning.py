import os
from sklearn.model_selection import GridSearchCV

class hypertuning():
    def __init__(self,model):
        self.train_X = model['train_X']
        self.train_y = model['train_y']
        self.model = model['model']
        self.param_grid = {
            'regressor__hidden_layer_sizes':[(10),(10,10),(20),(20,20),(30),(30,30)],
            'regressor__learning_rate': [
                'constant', 
                # 'invscaling', 
                'adaptive'
            ],
            'regressor__learning_rate_init': [0.001, 0.01, 0.1],
            'regressor__early_stopping': [False, True],
            'regressor__tol': [0.0001, 0.001, 0.01],
            'regressor__n_iter_no_change': [10, 20],
            # 'regressor__power_t': [0.5, 0.25],
            'regressor__warm_start':[False, True]
        }
        self.search_parameters = {
            'estimator':self.model,
            'cv':5,
            'scoring':"neg_root_mean_squared_error",
            'n_jobs':max(1,os.cpu_count()//4),
            'verbose':1
        }

    def tune(self):
        print('starting hypertuning')
        self.search_parameters['param_grid'] = self.param_grid
        grid_search = GridSearchCV(**self.search_parameters)
        grid_search.fit(self.train_X, self.train_y)
        print("Best Parameters:", grid_search.best_params_)
        return {key[2+key.find('__',0):]:value for key,value in grid_search.best_params_.items()}