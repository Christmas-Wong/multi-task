'''
Author: your name
Date: 2021-10-22 15:52:04
LastEditTime: 2021-10-24 22:08:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/utils/util_train.py
'''
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics_1(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_metrics_0(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=pred, average='binary')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def classification_report_csv(report, path):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2].strip())
        row['recall'] = float(row_data[3].strip())
        row['f1_score'] = float(row_data[4].strip())
        row['support'] = float(row_data[5].strip())
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(path, index = False)
