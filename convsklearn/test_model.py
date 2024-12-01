from model_settings import ms
from pathlib import Path
import pandas as pd
import os
from time import time
tic = time()
ms.find_root(Path())
models_dir = os.path.join(ms.root,ms.trained_models)
models = pd.Series([f for f in os.listdir(models_dir) if not f.startswith('.') and f.find('Legacy')])
for i,m in enumerate(models):
    print(f"{i}     {m}")

i = input("select model index: ")
selected_model = models[int(i)]

directory = os.path.join(models_dir,selected_model)

os.chdir(Path())
print(os.getcwd())

from .test import test
tester = test(directory=directory)
tester.load_model(verbose=True)
tester.plot_resutls()


runtime = time()-tic
print(f"cpu: {runtime}")