from relativize import unrelativize

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_RMSE(diff):
    if len(diff)>0:
        return np.sqrt(np.mean(diff.values**2))
        
def compute_MAE(diff):
    if len(diff)>0:
        return np.mean(np.abs(diff.values))


class test:
	def __init__(self,directory,retraining_frequency=20):
		self.retraining_frequency = retraining_frequency
		self.model_dir = directory

	def load_model(self,verbose=True):
		self.pickle = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')][0]
		self.picke_dir = os.path.join(self.model_dir,self.pickle)
		self.model = joblib.load(self.picke_dir)
		self.pricename = f"{self.model_dir[self.model_dir.rfind(' ')+1:]}_price"
		self.initial = self.model['model']
		if verbose!=False:
			print(self.initial)
		self.train_data = self.model['train_data'].copy()
		self.test_data = self.model['test_data'].copy()
		self.train_data['calculation_date'] = pd.to_datetime(self.train_data['calculation_date'],format='mixed')
		self.test_data['calculation_date'] = pd.to_datetime(self.test_data['calculation_date'],format='mixed')
		self.test_data = self.test_data.set_index('calculation_date').sort_index()
		self.train_data = self.train_data.set_index('calculation_date').sort_index()
		self.full_dataset = pd.concat([self.train_data,self.test_data])
		self.test_dates = self.test_data['date'].drop_duplicates().reset_index(drop=True)
		self.all_dates = self.full_dataset['date'].drop_duplicates().sort_values().reset_index(drop=True)
		self.all_dates = pd.to_datetime(self.all_dates,format='mixed')
		self.n = len(self.test_dates)//self.retraining_frequency

"""
usage
"""

from model_settings import ms
from pathlib import Path
import os
ms.find_root(Path())
models_dir = os.path.join(ms.root,ms.trained_models)
models = pd.Series([f for f in os.listdir(models_dir) if not f.startswith('.') and f.find('Legacy')])
for i,m in enumerate(models):
    print(f"{i}     {m}")
selected_model = models.iloc[2]
directory = os.path.join(models_dir,selected_model)


test = test(directory=directory)
test.load_model(verbose=True)



"""
sandbox
"""
print(test.test_data)

