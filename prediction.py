import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def ordinal_encoder(df, feats):
    for feat in feats:
        feat_val = list(np.arange(df[feat].nunique()))
        feat_key = list(df[feat].sort_values().unique())
        feat_dict = dict(zip(feat_key, feat_val))
        df[feat] = df[feat].map(feat_dict)
    return df


def get_prediction(data,model):
      
    return model.predict(data)

