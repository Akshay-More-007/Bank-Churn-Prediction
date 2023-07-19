import pickle
import pandas as pd
import numpy as np
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
<<<<<<< HEAD
model_path = os.path.join(curr_path, + '/app/model.pkl')
=======
model_path = os.path.join(curr_path, + 'model.pkl')
>>>>>>> fa4fc3789f3921fbdfc2344eb9a1dd64c4fb8cdc
model = pickle.load(open(model_path, 'rb'))

def predict_churn(attributes: np.ndarray):
    """ Returns 1 if customer is likely to churn, 0 otherwise"""
    # print(attributes.shape) # (1,10)

    pred = model.predict(attributes)
    print("Churn predicted")

    return int(pred[0])
