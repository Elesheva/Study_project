import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


#Confusion matrix
TP = 15
FN = 30
FP = 15
TN = 40 

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

F1 = 2 * (Precision * Recall) / (Precision + Recall)

#Nomber 1
df = pd.read_csv("C:/Users/vikto/Downloads/train_data_tree.csv")
X = df.drop(["num"], axis = 1)
y = df.num

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf.fit(X, y)

plt.figure(figsize=(60, 25))
tree.plot_tree(clf, fontsize=50, feature_names=list(X), filled=True)
plt.show()
