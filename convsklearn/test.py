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
	def __init__(self,directory):
		self.model_dir = directory

	def load_model(self):
		self.pickle = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')][0]
		self.picke_dir = os.path.join(self.model_dir,self.pickle)
		self.model = joblib.load(self.picke_dir)
		self.pricename = f"{self.model_dir[self.model_dir.rfind(' ')+1:]}_price"
		self.initial = self.model['model']
		print(self.initial)




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
selected_model = models.iloc[1]
directory = os.path.join(models_dir,selected_model)


test = test(directory=directory)
test.load_model()

"""
sandbox
"""
