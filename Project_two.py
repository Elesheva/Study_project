import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Дерево решений
data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})
clf = tree.DecisionTreeClassifier(criterion = "entropy")
X = data[["X_1", "X_2"]]
y = data.Y
clf.fit(X, y)
tree.plot_tree(clf, feature_names=list(X),
               class_names=['Negative', 'Positive'],
               filled=True)
plt.show()

#Энтропия и Information Gain(прирост информации)
# IG = E(Y) - E(Y/X)

# Шерстист
N = 10 
E_YX_SHERSTIT = (1/N) * 0 + (9/10) * 0.99
IG_SHERSTIT = 0.97 - E_YX_SHERSTIT

# Гавкает
E_YX_GAVKAET = (5/N) * 0 + (5/10) * 0.72
IG_GAVKAET = 0.97 - E_YX_GAVKAET

# Лазает
E_YX_LAZAET = (4/N) * 0 + (6/10) * 0
IG_LAZAET = 0.97 - E_YX_LAZAET

print(f"IG_Шерстист | IG_Гавкает | IG_Лазает\n{round(IG_SHERSTIT, 2)} {IG_GAVKAET} {IG_LAZAET}")

titanic = pd.read_csv('https://github.com/Ultraluxe25/Karpov-Stepik-Introduction-to-DS-and-ML/raw/main/csv/titanic.csv')
titanic.head()
titanic.columns

X = titanic.drop (columns = ["Cabin", "Ticket", 'Survived', 'Name', 'PassengerId'])
y = titanic.Survived
X = pd.get_dummies(X)
X = X.fillna({"Age": X.Age.median()})
X.head()
X.columns
clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 3)



X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.33, random_state= 1)

clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

plt.figure(figsize=(100, 25))
tree.plot_tree(clf, fontsize=10, feature_names=list(X), filled=True)
plt.show()

max_depth_p = range(1, 100)
scores_data = pd.DataFrame()

#КроссВалидация


for value in max_depth_p:
    clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = value)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    LL_cross_val_score = cross_val_score(clf, X_train, y_train, cv= 5).mean()
    temp_score_data = pd.DataFrame({"max_depth": [value],
                                    "train_score" : [train_score],
                                    "test_score": [test_score],
                                    "cross_val_score_e":[LL_cross_val_score]})
   
    scores_data = pd.concat([scores_data, temp_score_data])
    
scores_data.head()
scores_data_long = pd.melt(scores_data, id_vars = ["max_depth"], value_vars = ["train_score", "test_score", "cross_val_score_e"], var_name = "set_type", value_name = "score")

sns.lineplot(x = "max_depth", y = "score", hue = "set_type", data = scores_data_long  )
plt.show()

best_clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 10)
best_clf.fit(X_train, y_train)
best_clf.score(X_test, y_test)

#Задание 2

train = pd.read_csv("C:/Users/vikto/Downloads/train_iris.csv")
test = pd.read_csv("C:/Users/vikto/Downloads/test_iris.csv")

y_train = train.species
X_train = train.drop(['Unnamed: 0', "species"], axis = 1)

y_test  = test.species
X_test  = test.drop(['Unnamed: 0', "species"], axis = 1)

X_train.head()
y_train.head()
X_train.shape
y_train.shape

scores_data = pd.DataFrame()

value = range(1, 100)
np.random.seed(0)
for i in value:
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth = i)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    score_data = pd.DataFrame({"max_depth" : [i],
                               "train_score": [train_score],
                               "test_score" : [test_score]})
    scores_data = pd.concat([scores_data, score_data])

scores_data.head()

scores_data_long = pd.melt(scores_data, id_vars = ["max_depth"], value_vars = ["train_score", "test_score"], var_name = "set_type", value_name = "score")

sns.lineplot(x = "max_depth", y = "score", hue = "set_type", data = scores_data_long  )
plt.show()

#Задание 3

data = pd.read_csv("C:/Users/vikto/Downloads/dogs_n_cats.csv")
dogs_n_cats_test = pd.read_json("C:/Users/vikto/Downloads/dataset_209691_15.txt")
data.head()
data.isna().sum()
X = data.drop(["Вид"], axis = 1)
y = data["Вид"]
y = pd.get_dummies(y)
y = y.drop(["собачка"], axis = 1)
value = range(1, 100)
scores_data = pd.DataFrame()
for i in value:
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth = i)
    clf.fit(X, y)
    train_score = clf.score(X, y)
    #test_score = clf.score(X_test, y_test)
    cross_val_score_data = cross_val_score(clf, X, y, cv= 5).mean()
    score_data = pd.DataFrame({"max_depth" : [i],
                               "train_score": [train_score],
                               "cross_val_score":[cross_val_score_data]})
    scores_data = pd.concat([scores_data, score_data])

scores_data_long = pd.melt(scores_data, id_vars = ["max_depth"], value_vars = ["train_score", "cross_val_score"], var_name = "set_type", value_name = "score")
scores_data_long.head()

y_pred = pd.Series(clf.predict(dogs_n_cats_test), name='Вид')

# Подсчет значений в предсказанных данных
y_pred.value_counts()










