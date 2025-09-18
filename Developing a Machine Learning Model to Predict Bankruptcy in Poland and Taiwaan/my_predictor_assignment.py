

import gzip
import pandas as pd 
import json
import pickle

def wrangle(filepath):
    with gzip.open(filepath,"r")as f:
        taiwan_data=json.load(f)
    df=pd.DataFrame().from_dict(taiwan_data["observations"]).set_index("id")
    
    return df

def make_predictions(data_filepath, model_filepath):
    # Wrangle JSON file
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath,"rb")as f:
        model=pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred = pd.Series(y_test_pred,index=X_test.index,name="bankrupt")
    return y_test_pred
    