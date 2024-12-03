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
            target_name='observed_price',
            excluded_features=['barrier_price','asian_price','observed_price','outin','updown','n_fixings','barrier_cpu','asian_cpu'],
            seed=1312,
            ):
        self.seed = seed
        self.raw_data = pd.DataFrame()
        self.dataset = pd.DataFrame()
        self.target_name = target_name
        self.excluded_features = excluded_features
        self.numerical_features=[]
        self.categorical_features=[]
        self.feature_set=[]
        self.n_features=0
        self.development_dates = {}
        self.test_dates = {}
        self.train_data = {}
        self.test_data = {}
        self.train_X = {}
        self.train_y = {}
        self.test_X = {}
        self.test_y = {}
        self.preprocessor = None
        self.model = None
        self.fitted = None
        self.runtime = 0
        self.numerical_scaler = StandardScaler()
        self.mlp_params = {
            'activation':'relu',
            'hidden_layer_sizes': (10,),
            'max_iter': 500
        }

    def load_data(self,data):
        self.raw_data = data
        self.dataset = self.raw_data.dropna().copy()
        self.dataset['calculation_date'] = pd.to_datetime(self.dataset['calculation_date'],format='mixed',errors='coerce')
        self.dataset['date'] = pd.to_datetime(self.dataset['date'],format='mixed',errors='coerce')
        features = self.dataset.dtypes.reset_index(drop=False)

        self.numerical_features = features[
            (features[features.columns[-1]]==int)
            |(features[features.columns[-1]]==float)
        ].iloc[:,0].tolist()

        self.numerical_features = [f for f in self.numerical_features if f not in self.excluded_features]

        self.categorical_features = features[
            (features[features.columns[-1]]==object)
        ].iloc[:,0].tolist()

        self.categorical_features = [f for f in self.categorical_features if f not in self.excluded_features]

        self.feature_set = self.numerical_features + self.categorical_features
        
        self.n_features = len(self.feature_set)
        
        if self.seed != None:
            self.mlp_params['random_state'] = self.seed

        self.transformers = [
            ("StandardScaler",self.numerical_scaler,self.numerical_features),
            ("OneHotEncoder", OneHotEncoder(sparse_output=False),self.categorical_features)
        ]

    """
    ===========================================================================
    preprocessing
    """


    def preprocess_data(self, development_dates, test_dates,plot=True):
        self.development_dates = development_dates
        self.test_dates = test_dates
        development_dates_dt = pd.to_datetime(self.development_dates, errors='coerce')
        test_dates_dt = pd.to_datetime(self.test_dates, errors='coerce')

        try:
            self.dataset['date'] = pd.to_datetime(self.dataset['date'], format='mixed', errors='coerce')
            self.train_data = self.dataset[self.dataset['date'].isin(development_dates_dt)].sort_values(by='date')
            self.test_data = self.dataset[self.dataset['date'].isin(test_dates_dt)].sort_values(by='date')

        except Exception:
            self.dataset['calculation_date'] = pd.to_datetime(self.dataset['calculation_date'], format='mixed', errors='coerce')
            self.train_data = self.dataset[self.dataset['calculation_date'].isin(development_dates_dt)].sort_values(by='calculation_date')
            self.test_data = self.dataset[self.dataset['calculation_date'].isin(test_dates_dt)].sort_values(by='calculation_date')

        self.test_X = self.test_data[self.feature_set]
        self.test_y = self.test_data[self.target_name]
        self.train_X = self.train_data[self.feature_set]
        self.train_y = self.train_data[self.target_name]
        self.preprocessor = ColumnTransformer(transformers=self.transformers)

        if plot == True:
            trainplotx = pd.date_range(start=min(self.development_dates),end=max(self.development_dates),periods=self.train_data.shape[0])
            testplotx = pd.date_range(start=min(self.test_dates),end=max(self.test_dates),periods=self.test_data.shape[0])
            plt.figure()
            plt.plot(testplotx,self.test_data['spot_price'].values,color='purple',label='out-of-sample')
            plt.plot(trainplotx,self.train_data['spot_price'].values,color='green',label='in-sample')
            plt.xticks(rotation=45)
            plt.legend(loc='upper left')
            plt.show()



    """
    ===========================================================================
    model estimation
    """

    def fit_scaled_target_mlp(self, print_details=True):
        if print_details == True:
            print(f'\ntraining on {self.train_X.shape[0]} samples...\n')
            for p,v in self.mlp_params.items():
                print(f"{p}: {v}")
        dnn_start = time.time()
        self.regressor = MLPRegressor(**self.mlp_params)
                                  
        self.dnn_pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", self.regressor)
        ])
        
        self.model = TransformedTargetRegressor(
            regressor=self.dnn_pipeline,
            transformer=self.numerical_scaler
        )
        
        self.fitted = self.model.fit(self.train_X,self.train_y.values)
        dnn_end = time.time()
        self.runtime = dnn_end - dnn_start
        if print_details==True:
            print(f"cpu: {self.runtime}")

    def construct_mlp(self):
        self.regressor = MLPRegressor(**self.mlp_params)         
        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", self.regressor)
        ])

    def fit_mlp(self, print_details=True):
        if print_details == True:
            print(f'\ntraining on {self.train_X.shape[0]} samples...\n')
            for p,v in self.mlp_params.items():
                print(f"{p}: {v}")
        self.mlp_start = time.time()
        self.fitted = self.model.fit(self.train_X,self.train_y.values)
        self.mlp_end = time.time()
        self.runtime = self.mlp_end - self.mlp_start
        print(f"cpu: {self.runtime}")
    """
    ===========================================================================
    standard model testing
    """
    
    def test_prediction_accuracy(self):
        
        insample_prediction = np.maximum(self.fitted.predict(self.train_X),0)
        insample_diff = insample_prediction - self.train_y
        insample_RMSE = np.sqrt(np.average(insample_diff**2))
        insample_MAE = np.average(np.abs(insample_diff))
        
        outofsample_prediction = np.maximum(self.fitted.predict(self.test_X),0)
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
