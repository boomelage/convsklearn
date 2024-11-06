#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

a proprietary class of convenience wrappers for sklearn


"""
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class convsklearn:
    """

    a proprietary class of convenience wrappers for sklearn


    """
    def __init__(
            self,
            target_name=None,
            numerical_features=None,
            categorical_features=None
            ):
        if target_name is not 'target_name':
            self.target_name = target_name
        else:
            self.target_name = 'target_name'

        if numerical_features is not None:
            self.numerical_features = numerical_features
        else:
            self.numerical_features = []

        if categorical_features is not None:
            self.categorical_features = categorical_features
        else:
            self.categorical_features = []

        if len(self.numerical_features)+len(self.categorical_features)>0:
            self.feature_set = self.numerical_features + self.categorical_features
            self.n_features = len(self.feature_set)
        else:
            self.feature_set = []
            self.n_features = 0

        self.dnn_params = {
            'alpha': 0.01, 
            'hidden_layer_sizes': (self.n_features, self.n_features), 
            'learning_rate': 'adaptive', 
            'learning_rate_init': 0.1, 
            'solver': 'sgd',
            'early_stopping': False, 
            'max_iter': 500,
            'warm_start': True,
            'tol': 0.0001
        }
        
        self.transformers = [
            ("StandardScaler",StandardScaler(),self.numerical_features),
            ("OneHotEncoder", OneHotEncoder(
                sparse_output=False),self.categorical_features)
        ]

        self.target_transformer_pipeline = Pipeline([
                ("StandardScaler", StandardScaler()),
                ])
        self.train_data = {}
        self.test_data = {}
        self.train_X = {}
        self.train_y = {}
        self.test_X = {}
        self.test_y = {}
        self.preprocessor = None
        self.pipeline = None
        self.model = None
        self.model_fit = None
        self.dnn_runtime = 0


    """            
    ===========================================================================
    preprocessing
    """

    def get_train_test_arrays(self,
            train_data, test_data,
            feature_set=None, target_name=None
            ):
        
        if feature_set == None:
            feature_set = self.feature_set
            if len(feature_set)==0:
                raise('no feautres specified')
        if target_name == None:
            target_name = self.target_name
            if target_name == None:
                raise('no target specified')

        test_X = test_data[feature_set]
        test_y = test_data[target_name]
        train_X = train_data[feature_set]
        train_y = train_data[target_name]
        return {
            'train_X':train_X, 
            'train_y':train_y, 
            'test_X':test_X, 
            'test_y':test_y
        }


    def preprocess_data(self,dataset,development_dates,test_dates):
        self.dataset = dataset
        self.development_dates = development_dates
        self.test_dates = test_dates
        try:
            self.train_data = dataset[dataset['date'].isin(development_dates)].sort_values(by='date')
            self.test_data = dataset[dataset['date'].isin(test_dates)].sort_values(by='date')
        except Exception:
            self.train_data = dataset[dataset['calculation_date'].isin(development_dates)].sort_values(by='calculation_date')
            self.test_data = dataset[dataset['calculation_date'].isin(test_dates)].sort_values(by='calculation_date')

        trainplotx = pd.date_range(start=min(self.development_dates),end=max(self.development_dates),periods=self.train_data.shape[0])
        testplotx = pd.date_range(start=min(self.test_dates),end=max(self.test_dates),periods=self.test_data.shape[0])

        plt.figure()
        plt.plot(testplotx,self.test_data['spot_price'].values,color='purple',label='out-of-sample')
        plt.plot(trainplotx,self.train_data['spot_price'].values,color='green',label='in-sample')
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.show()
        arrs = self.get_train_test_arrays(
        self.train_data, self.test_data)
        self.train_X = arrs['train_X']
        self.train_y = arrs['train_y']
        self.test_X = arrs['test_X']
        self.test_y = arrs['test_y']
        self.preprocessor = ColumnTransformer(transformers=self.transformers)

        return {
            'preprocessor':self.preprocessor,
            'train_X' : self.train_X,
            'train_y':self.train_y,
            'test_X':self.test_X,
            'test_y':self.test_y,
            'train_data':self.train_data,
            'test_data':self.test_data
        }
    
    """
    ===========================================================================
    model estimation
    """

    def run_dnn(self, print_details=True):
        if print_details == True:
            print('\ntraining...\n')
            for p,v in self.dnn_params.items():
                print(f"{p}: {v}")
        dnn_start = time.time()
        self.regressor = MLPRegressor(**self.dnn_params)
                                  
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", self.regressor)
        ])
        
        self.model = TransformedTargetRegressor(
            regressor=self.pipeline,
            transformer=self.target_transformer_pipeline 
        )
        
        self.model_fit = self.model.fit(self.train_X,self.train_y)
        dnn_end = time.time()
        self.dnn_runtime = dnn_end - dnn_start
        if print_details==True:
            print(f"cpu: {self.dnn_runtime}")
        return self.model_fit

    """
    ===========================================================================
    standard model testing
    """
    
    def test_prediction_accuracy(self):
        
        insample_prediction = np.maximum(model_fit.predict(self.train_X),0)
        insample_diff = insample_prediction - self.train_y
        insample_RMSE = np.sqrt(np.average(insample_diff**2))
        insample_MAE = np.average(np.abs(insample_diff))
        
        outofsample_prediction = np.maximum(self.model_fit.predict(self.test_X),0)
        outofsample_diff = outofsample_prediction-self.test_y
        outofsample_RMSE = np.sqrt(np.average(outofsample_diff**2))
        outofsample_MAE = np.average(np.abs(outofsample_diff))
        
        print("\nin sample:"
              f"\n     RMSE: {insample_RMSE}"
              f"\n     MAE: {insample_MAE}")
        print("\nout of sample:"
              f"\n     RMSE: {outofsample_RMSE}"
              f"\n     MAE: {outofsample_MAE}")
        self.train_data['insample_target'] = self.train_y
        self.train_data['insample_prediction'] = insample_prediction 
        self.train_data['insample_error'] = insample_diff 
        
        self.test_data['outofsample_target'] = self.test_y
        self.test_data['outofsample_prediction'] = outofsample_prediction
        self.test_data['outofsample_error'] = outofsample_diff
        
        return {'train_data':self.train_data,'test_data':self.test_data}
