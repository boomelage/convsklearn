from relativize import unrelativize

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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
		self.wdir = Path()


	def load_model(self,verbose=True):
		self.pickle = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')][0]
		self.picke_dir = os.path.join(self.model_dir,self.pickle)
		self.model = joblib.load(self.picke_dir)
		self.pricename = f"{self.model_dir[self.model_dir.rfind(' ')+1:]}_price"
		self.initial = self.model['model']
		if verbose!=False:
			print(self.initial)
		self.train_data = unrelativize(self.model['train_data'].copy())
		self.test_data = unrelativize(self.model['test_data'].copy())
		self.train_data['calculation_date'] = pd.to_datetime(self.train_data['calculation_date'],format='mixed')
		self.test_data['calculation_date'] = pd.to_datetime(self.test_data['calculation_date'],format='mixed')
		self.test_data = self.test_data.set_index('calculation_date').sort_index()
		self.train_data = self.train_data.set_index('calculation_date').sort_index()
		self.test_dates = self.test_data['date'].drop_duplicates().reset_index(drop=True)
		self.n = len(self.test_dates)//self.retraining_frequency
		self.dump = os.path.join(self.wdir,f"{self.pricename[:self.pricename.find('_')]} analysis")
		if not os.path.exists(self.dump):
			os.mkdir(self.dump)
		os.chdir(self.dump)

	def split_quantiles(self):
		self.lower = self.test_data[self.test_data['outofsample_error']<self.test_data['outofsample_error'].describe()['50%']].copy()
		self.upper = self.test_data[self.test_data['outofsample_error']>self.test_data['outofsample_error'].describe()['50%']].copy()

	def plot_pairs(self):
		pairplot_upper = sns.pairplot(self.upper[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(r"lower.png", dpi=600)
		print('saved lower')
		pairplot_lower = sns.pairplot(self.lower[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(os.path.abspath(r"upper.png"), dpi=600)
		print('saved upper')

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
selected_model = models.iloc[0]
directory = os.path.join(models_dir,selected_model)



os.chdir(ms.root)
print(os.getcwd())

tester = test(directory=directory)
tester.load_model(verbose=True)
tester.split_quantiles()
tester.plot_pairs()

"""
sandbox
"""





# cols = ['outofsample_MAE','outofsample_RMSE']
# df = pd.DataFrame()
# models = {}

# for i in range(0,test.n):
#     subset_test_dates = pd.to_datetime(test.model['test_dates'][(i*test.retraining_frequency):(i+1)*test.retraining_frequency],format='fixed')
#     subset_test = test.test_data[test.test_data['date'].isin(subset_test_dates)]
    
#     target = subset_test['relative_observed']
#     prediction = test.initial.predict(subset_test[test.model['feature_set']])
    
#     error = prediction-target
    
#     predicted_price = prediction*subset_test['strike_price']
#     pricing_error = prediction-subset_test[test.pricename]
    
#     date = subset_test_dates.iloc[0]
#     df.at[date,'outofsample_MAE'] = compute_MAE(error)
#     df.at[date,'outofsample_RMSE'] = compute_RMSE(error)
#     df.at[date,'avgsqrtv0'] = np.mean(np.sqrt(subset_test['v0']))
#     for col in [
#         'rho','theta',
#         'spot_price'
#     ]:
#         df.at[date,f"avg_{col}"] = np.mean(subset_test[col])

# df.index = pd.to_datetime(df.index)
# print(df)