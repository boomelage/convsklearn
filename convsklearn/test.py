from relativize import unrelativize

import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay
import plotly.express as px


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
		self.security_tag = self.pricename[:self.pricename.find('_')].title()+' Options'
		self.dump = os.path.join(self.wdir,f"{self.pricename[:self.pricename.find('_')]} analysis")
		if not os.path.exists(self.dump):
			os.mkdir(self.dump)
		os.chdir(self.dump)
		self.lower = self.test_data[self.test_data['outofsample_error']<self.test_data['outofsample_error'].describe()['50%']].copy()
		self.upper = self.test_data[self.test_data['outofsample_error']>self.test_data['outofsample_error'].describe()['50%']].copy()

	def plot_pairs(self):
		pairplot_upper = sns.pairplot(self.upper[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(r"pairs_lower.png", dpi=600)
		print('saved lower')
		pairplot_lower = sns.pairplot(self.lower[['kappa','theta','rho','eta','v0','outofsample_error']])
		plt.savefig(os.path.abspath(r"pairs_upper.png"), dpi=600)
		print('saved upper')

	def plot_dists(self):
		sns.kdeplot(data=self.test_data, x='observed_price', label='Estimated', color='purple')
		sns.histplot(data=self.test_data, x=self.pricename, label='Target', color='green', stat='density', alpha=0.5)
		plt.legend()
		plt.savefig(r"price_dist.png")
		plt.close()
		train_zoom = self.test_data[
		    (self.test_data['relative_observed']>0.05)
		    &(self.test_data['relative_observed']<0.5)
		]
		sns.kdeplot(data=train_zoom, x='observed_price', label='Estimated', color='purple')
		sns.histplot(data=train_zoom, x=self.pricename, label='Target', color='green', stat='density', alpha=0.5)
		plt.legend()
		plt.savefig(r"price_dist_zoom.png")
		plt.close()

	def plot_importances(self):
		r = permutation_importance(self.initial, self.train_data[self.model['feature_set']], self.train_data[self.model['target_name']],
                           n_repeats=30,
                           random_state=1312,
                           scoring='neg_mean_squared_error'
                          )
		importances = pd.DataFrame(data=r['importances'],index=self.model['feature_set']).T
		importances_mean = pd.Series(r['importances_mean'],index=self.model['feature_set'])
		importances_std = pd.Series(r['importances_std'],index=self.model['feature_set'])
		fig = px.box(
		    importances[self.model['feature_set']],
		    height=1000,
		    width=1200,
		    facet_col_spacing=0,
		    facet_row_spacing=0,
		    notched=True,
		    title=f'Feature Importance for {self.security_tag}'
		)
		fig.update_xaxes(title='Feature')
		fig.update_yaxes(title='')
		fig.write_image("feature_importance.png", scale=6.25)

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
		display.figure_.savefig("partial_dependence.png", dpi=600)

	def plot_resutls(self):
		self.plot_dists()
		self.plot_dependancies()
		self.plot_pairs()

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
tester.plot_resutls()


"""
sandbox
"""

