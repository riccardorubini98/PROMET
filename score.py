from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

def get_score(y_true, y_pred, labels_names):
    """ 
    return:
        - accuracy score
        - f1 micro score
        - pd df with metrics for each type
    """
    # one_hot -> label_idx
    y_true = y_true.argmax(dim=1)
    y_pred = y_pred.argmax(dim=1)
    # label_idx -> label_name
    y_true = [labels_names[pred] for pred in y_true]
    y_pred = [labels_names[pred] for pred in y_pred]
    # back to one_hot
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(y_true)
    y_pred = mlb.transform(y_pred)
    # get metrics
    acc = round(accuracy_score(y_true, y_pred),4)*100
    f1_micro = round(f1_score(y_true, y_pred, average='micro', zero_division=0),4)*100
    # get report
    report = classification_report(y_true, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'full_type'})
    df[['avoid', 'father_type', 'son_type']] = df['full_type'].str.split("/", expand = True)
    df = df.drop('avoid', axis=1)
    return acc, f1_micro, df