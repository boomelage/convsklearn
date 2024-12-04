import joblib
import os
def load_model(models_dir):
    if not os.path.exists(models_dir):
        models_dir = os.path.join(ms.root,ms.MacDirEx,ms.trained_models)
    models = os.listdir(models_dir)
    for i,m in enumerate(models):
        print(f"{i}   {m}")
    i = int(input('select model: '))
    selected_model = models[i]
    model_dir = os.path.join(models_dir,selected_model,selected_model+'.pkl')
    return joblib.load(model_dir)