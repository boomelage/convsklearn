from .train import train
import os
from time import time
from model_settings import ms
from pathlib import Path
from df_collector import df_collector

start = time()
ms.find_root(Path())
df_collector.root = ms.root
raw = df_collector.cboe_spx_barriers().iloc[:,1:]
train = train()
train.load_data(raw,verbose=True)
train.construct(verbose=True,plot=False)
train.fit()
train.test_fit()
train.save_model(dir=os.path.join(ms.root,ms.trained_models))
runtime = time() - start
print(f"cpu: {runtime}")