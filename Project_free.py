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

#RANDOM FOREST

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

 
titanic_data = pd.read_csv('C:/Users/vikto/Downloads/titanic.csv')
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived
X = pd.get_dummies(X)
X = X.fillna({'Age': X.Age.median()})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
clf_rf = RandomForestClassifier()
parametrs = {"n_estimators" : [10, 20, 30], "max_depth" : [2, 5, 7, 10]}
grid_searh_cv_clf = GridSearchCV(clf_rf, parametrs, cv = 5)
grid_searh_cv_clf.fit(X_train, y_train)
grid_searh_cv_clf.best_params_
best_clf = grid_searh_cv_clf.best_estimator_
best_clf.score(X_test, y_test)
feature_importances = best_clf.feature_importances_

feature_importances_df = pd.DataFrame({"features": list(X_train),"feature_importances": feature_importances})

feature_importances_df.sort_values("feature_importances", ascending=False)

#ЗАДАНИЕ 1
df =  pd.read_csv("C:/Users/vikto/Downloads/heart-disease.csv")
X_train = df.loc[:,"age":"thal"]
y_train = df["target"]

np.random.seed(0)

rf = RandomForestClassifier(10, max_depth=5)
rf.fit(X_train, y_train)
imp = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()

#ЗАКЛЮЧЕНИЕ 
#1
df = pd.read_csv("C:/Users/vikto/Downloads/training_mush.csv")
X = df.drop(["class"], axis = 1)
y = df["class"]
clf_rf = RandomForestClassifier(random_state=0)


parametrs = {'n_estimators':range(10,50,10),
'max_depth':range(1,12,2),
'min_samples_leaf':range(1,7),
'min_samples_split':range(2,9,2)}


grid_searh_cv = GridSearchCV(clf_rf, parametrs, cv = 3)

grid_searh_cv.fit(X, y)

best_clf = grid_searh_cv.best_estimator_
feature_importances = best_clf.feature_importances_


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(
    data=imp.reset_index(),
    x="importance",
    y="index",
    order=imp.sort_values("importance", ascending=False).index
)
plt.title("Feature Importances")
plt.show()

test = pd.read_csv("C:/Users/vikto/Downloads/testing_mush.csv")

pred = best_clf.predict(test)
df_pred = pd.DataFrame(pred)
df_pred.columns
df_pred["predictions"] = df_pred[:]
count = df_pred.predictions.sum()
df_pred = df_pred.drop([0],axis = 1)

new_data = pd.read_csv("C:/Users/vikto/Downloads/testing_y_mush.csv/testing_y_mush.csv")

from sklearn.metrics import confusion_matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(new_data, df_pred), annot=True, cmap="Blues")
plt.title("Корреляция признаков")
plt.show()

#NEXT_nomber, zykesы

train = pd.read_csv("C:/Users/vikto/Downloads/invasion.csv")
test = pd.read_csv("C:/Users/vikto/Downloads/operative_information.csv")
X_train = train.drop(["class"], axis = 1)
y_train = train["class"]

clf_rf = RandomForestClassifier()

parametrs = {'n_estimators':range(10,50,10),
'max_depth':range(1,12,2),
'min_samples_leaf':range(1,7),
'min_samples_split':range(2,9,2)}

grid_searh_cv = GridSearchCV(clf_rf, parametrs, cv = 3)

grid_searh_cv.fit(X_train, y_train)
best_clf = grid_searh_cv.best_estimator_
pred = best_clf.predict(test)
fighter = 0
transport = 0
cruiser = 0

for i in range(len(pred)):
    if pred[i] == "fighter":
        fighter = fighter + 1
    elif pred[i] == "transport":
        transport = transport + 1
    else:
        cruiser = cruiser + 1

print("fighter:", fighter)
print("transport:", transport)
print("cruiser:", cruiser)



feature_importances = best_clf.feature_importances_

imp = pd.DataFrame(feature_importances, index=X_train.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(12, 8))
plt.show()

cosmos = pd.read_csv("C:/Users/vikto/Downloads/space_can_be_a_dangerous_place.csv")

cosmos.corr().sort_values(["dangerous"], ascending = False)








