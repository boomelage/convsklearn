from .relativize import relativize
from .convsklearn import convsklearn
from .hypertuning import hypertuning

import os
import joblib
from datetime import datetime
from time import time
import numpy as np

def noisify(x):
    return x + np.random.normal(scale=x*0.01)

class train:
	"""


	example usage:

import os
from model_settings import ms
from pathlib import Path
from df_collector import df_collector
ms.find_root(Path())
df_collector.root = ms.root
raw = df_collector.cboe_spx_asians().iloc[:,1:]
train = train()
train.load_data(raw,verbose=False)
train.construct(verbose=False,plot=False)
train.fit()
train.test_fit()
train.save_model(dir=os.path.join(ms.root,ms.trained_models))



	"""
	def __init__(self):
		self.data = {}

	def load_data(self,data,verbose=True):
		self.data = data
		self.pricename = [f for f in self.data.columns if f.find('_price')!=-1 and f.find('spot_')==-1 and f.find('strike_')==-1][0]
		self.relative_pricename = 'relative_'+self.pricename
		self.data = self.data[self.data[self.pricename]<=self.data['spot_price']]
		self.data = relativize(self.data)
		self.filetag = f'cboe spx relative {self.pricename[:self.pricename.find('_',0)]}'
		self.relative_observed = self.data[self.relative_pricename].values
		self.relative_observed[self.relative_observed>0] = noisify(self.relative_observed[self.relative_observed>0])
		self.targetname = 'relative_observed'
		self.data[self.targetname]= self.relative_observed
		print(f'collected {self.pricename[:self.pricename.find('_',0)]} options')
		if verbose != False:
			print(self.data.describe())
			print(self.data.dtypes)
			print(self.data['calculation_date'].drop_duplicates().reset_index(drop=True))
			print('\n')

	def construct(self,verbose=True,plot=True):
		self.trainer = convsklearn()
		self.trainer.target_name = self.targetname
		self.trainer.excluded_features = self.trainer.excluded_features + \
			['spot_price','strike_price','barrier','rebate',self.relative_pricename,'relative_observed']
		self.trainer.load_data(self.data)
		if verbose != False: 
			print('features:')
			for f in self.trainer.feature_set:
			    print(f"   {f}")
			print(f"\ntarget:\n   {self.trainer.target_name}\n")
		self.dates = self.data['date'].drop_duplicates()
		self.development_dates = self.dates[:100]#len(dates)//3]
		self.test_dates = self.dates[~self.dates.isin(self.development_dates)]
		self.trainer.preprocess_data(self.development_dates,self.test_dates,plot=plot)
		self.trainer.construct_mlp()
		if verbose != False:
			print('instance variables:')
			for key, value in self.trainer.__dict__.items():
				print(f"{key}:\n{value}\n")

	def fit(self):
		self.m = {'train_X':self.trainer.train_X,'train_y':self.trainer.train_y,'model':self.trainer.model}
		self.hyper = hypertuning(self.m)
		self.trainer.mlp_params.update(self.hyper.tune())
		self.trainer.construct_mlp()
		print(self.trainer.model)
		self.trainer.fit_mlp()

	def test_fit(self):
		self.train_test = self.trainer.test_prediction_accuracy()

	def save_model(self,dir):
		train_end_tag = datetime.fromtimestamp(time()).strftime(r'%Y-%m-%d %H%M%S%f')
		self.file_tag = str(train_end_tag + " inital " + self.filetag)
		files_dir = os.path.join(dir,self.file_tag)
		if os.path.exists(files_dir):
			pass
		else:
			os.mkdir(files_dir)
		file_dir = os.path.join(files_dir,self.file_tag)
		joblib.dump(self.trainer.__dict__,str(f"{file_dir}.pkl"))
		print(f'\nmodel saved to {file_dir}')

