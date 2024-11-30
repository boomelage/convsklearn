from relativize import relativize
from convsklearn import convsklearn

import numpy as np

def noisify(x):
    return x + np.random.normal(scale=x*0.01)

class train:
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

	def construct(self,verbose=True):
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


from model_settings import ms
from pathlib import Path
from df_collector import df_collector
ms.find_root(Path())
df_collector.root = ms.root
raw = df_collector.cboe_spx_barriers().iloc[:,1:]



train = train()
train.load_data(raw,verbose=False)
train.construct(verbose=True)