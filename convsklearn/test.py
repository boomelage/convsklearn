from .relativize import unrelativize

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

def compute_RMSE(diff):
    if len(diff)>0:
        return np.sqrt(np.mean(diff.values**2))
        
def compute_MAE(diff):
    if len(diff)>0:
        return np.mean(np.abs(diff.values))


class test:
	"""

	example usage


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
tester.plot_resutls()

	"""

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
		self.security_tag = self.pricename[:self.pricename.find('_')].title()+' Options'
		self.file_tag = f"{self.pricename[:self.pricename.find('_')]} analysis"
		self.dump = os.path.join(self.wdir,self.file_tag)
		if not os.path.exists(self.dump):
			os.mkdir(self.dump)
		os.chdir(self.dump)
		self.lower = self.test_data[self.test_data['outofsample_error']<self.test_data['outofsample_error'].describe()['50%']].copy()
		self.upper = self.test_data[self.test_data['outofsample_error']>self.test_data['outofsample_error'].describe()['50%']].copy()

	def plot_pairs(self):
		pairplot_upper = sns.pairplot(self.upper[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(os.path.abspath(f"{self.file_tag} pairs_lower.png"), dpi=600)
		plt.close()
		print('saved lower')
		pairplot_lower = sns.pairplot(self.lower[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(os.path.abspath(f"{self.file_tag} pairs_upper.png"), dpi=600)
		print('saved upper')
		plt.close()

	def plot_dists(self):
		sns.kdeplot(data=self.test_data, x='observed_price', label='Estimated', color='purple')
		sns.histplot(data=self.test_data, x=self.pricename, label='Target', color='green', stat='density', alpha=0.5)
		plt.legend()
		plt.savefig(f"{self.file_tag} price_dist.png")
		plt.close()
		train_zoom = self.test_data[
		    (self.test_data['relative_observed']>0.05)
		    &(self.test_data['relative_observed']<0.5)
		]
		sns.kdeplot(data=train_zoom, x='observed_price', label='Estimated', color='purple')
		sns.histplot(data=train_zoom, x=self.pricename, label='Target', color='green', stat='density', alpha=0.5)
		plt.legend()
		plt.savefig(f"{self.file_tag} price_dist_zoom.png")
		plt.close()
		print('saved price distributions')

	def plot_importances(self):
		r = permutation_importance(self.initial, self.train_data[self.model['feature_set']], self.train_data[self.model['target_name']],
                           n_repeats=30,
                           random_state=1312,
                           scoring='neg_mean_squared_error'
                          )
		importances = pd.DataFrame(data=r['importances'],index=self.model['feature_set']).T
		importances_mean = pd.Series(r['importances_mean'],index=self.model['feature_set'])
		importances_std = pd.Series(r['importances_std'],index=self.model['feature_set'])
		print('importances computed')
		plt.figure(figsize=(12, 10))  # Set figure size
		sns.boxplot(
		    data=importances[self.model['feature_set']],
		    notch=True
		)
		plt.title(f'Feature Importance for {self.security_tag}', fontsize=16)
		plt.xlabel('Feature', fontsize=14)
		plt.ylabel('', fontsize=14)
		plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
		plt.tight_layout()  # Adjust layout to prevent overlap
		plt.savefig(f"{self.file_tag} feature_importance.png", dpi=625)  # Save the figure
		plt.close()
		print('saved importances')

	def plot_dependancies(self):
		common_params = {
			"subsample": 50,
			"n_jobs": 2,
			"grid_resolution": 20,
			"random_state": 0,
		}
		PDPfeatures = [f for f in self.model['numerical_features']]
		features_info = {
			"features": PDPfeatures,
			"kind": "average",
			"categorical_features":self.model['categorical_features']
		}
		_, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
		display = PartialDependenceDisplay.from_estimator(
			self.initial,
			self.model['train_X'],
			**features_info,
			ax=ax,
			**common_params,
		)
		_ = display.figure_.suptitle(
			(
				f"Partial dependence for {self.security_tag}"
			),
			fontsize=16,
		)
		display.figure_.savefig(f"{self.file_tag} partial_dependence.png", dpi=600)
		plt.close()
		print('saved dependencies')

	def plot_resutls(self):
		self.plot_importances()
		self.plot_dists()
		self.plot_pairs()
		self.plot_dependancies()
		

