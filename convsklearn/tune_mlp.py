import os
import joblib
from pathlib import Path
from model_settings import ms
from convsklearn.hypertuning import hypertuning

class tune_mlp():
	def __init__(self):
		ms.find_root(Path())
		self.models_dir = os.path.join(ms.root,ms.trained_models)
		self.models = [f for f in os.listdir(self.models_dir) if f.find('Legacy')==-1 and not f.startswith('.')]
		self.param_grid = {
		    'regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
		    'regressor__learning_rate_init': [0.001, 0.01, 0.1],
		    'regressor__early_stopping': [False, True],
		    'regressor__tol': [0.0001, 0.001, 0.01],
		    'regressor__n_iter_no_change': [10, 20],
		    'regressor__power_t': [0.5, 0.25],
		}
		for i,m in enumerate(self.models):
			print(f"{i}   {m}")

	def tune(self, index):
		self.model_dir = os.path.join(self.models_dir,self.models[index])
		self.pricename = self.model_dir[self.model_dir.rfind(' ',0)+1:]+"_price"
		self.pickle = [os.path.join(self.model_dir,f) for f in os.listdir(self.model_dir) if f.endswith('.pkl')][0]
		self.model = joblib.load(self.pickle)
		self.mlp = self.model['model']
		hyper = hypertuning(self.model)
		hyper.param_grid = self.param_grid
		print(hyper.search_parameters)
		self.new = hyper.tune()
		self.model['mlp_params'].update(self.new)
		self.tuned_params = self.model['mlp_params']
		param_dump = os.path.join(ms.root,Path(ms.trained_models).parent,'tuned_parameters')
		if not os.path.exists(param_dump):
		    os.mkdir(param_dump)
		with open(os.path.join(param_dump,f'{self.pricename[:self.pricename.find('_')]}_parameters.py'), 'w') as f:
		    f.write(f"tuned_params = {repr(self.tuned_params)}")