import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/vikto/Downloads/StudentsPerformance.csv")
students_performance = pd.read_csv("C:/Users/vikto/Downloads/StudentsPerformance.csv")
df.head()


#Фильтрация данных
df.gender == "female"
df.loc[df.gender == "female", ["gender", "writing score"]]
mean_writing_score = df['writing score'].mean()
df.loc[df['writing score'] > mean_writing_score]
students_performance[(students_performance['writing score']) > 100 | (students_performance.gender == 'female')]
student_lunch =df.loc[df.lunch == "free/reduced",["writing score", "math score", "reading score"]].describe()
student_standart =df.loc[df.lunch == "standard",["writing score", "math score", "reading score"]].describe()
students_performance = students_performance.rename(columns = {"parental level of education": "parental_level_of_education", "test preparation course" : "test_preparation_course", "math score": "math_score", "reading score" : "reading_score", "writing score" : "writing_score"})
lolkek = 74
students_performance.query("gender == 'female' & writing_score > @lolkek ")
students_performance.filter(like= "score")


#Группировка и агрегация
students_performance.groupby("gender", as_index = False).aggregate({"math_score" : "mean", "reading_score" : "mean"})
students_performance.groupby(["gender", "race/ethnicity"]).math_score.nunique()
students_performance.sort_values(["gender", "race/ethnicity"], ascending = False) \
    .groupby("gender").head(5)
students_performance = students_performance.assign(total_score_log = np.log(students_performance.math_score))
students_performance.drop(["total_score_log"], axis = 1)

dota2 = pd.read_csv("C:/Users/vikto/Downloads/dota_hero_stats.csv")
cc = dota2.groupby(["attack_type", "primary_attr"], as_index = False).count()


pupa_lupa= pd.read_csv("C:/Users/vikto/Downloads/accountancy.csv")
pupa_or_lupa = pupa_lupa.groupby(["Type", "Executor"]).aggregate({"Salary" : "mean"}).sort_values("Type", ascending = False)

vodorosly = pd.read_csv("C:/Users/vikto/Downloads/algae.csv")
alanin = vodorosly.loc[vodorosly.genus == "Fucus", ["alanin"]]
alanin.mean()
group = vodorosly.groupby('group').aggregate({'glucose': lambda x: max(x) - min(x) })


#Визуализация, seaborn
df = pd.read_csv("C:/Users/vikto/Downloads/income.csv")
import matplotlib.pyplot as plt
import seaborn as sns 

sns.lmplot(x = "math_score", y = "reading_score", hue= "gender", data= students_performance, fit_reg= False)

sns.lineplot(x = df.index, y = df.income)
sns.lineplot(data= df)
plt.show()
df.plot(kind = "line")
df.income.plot()
df.scatterplot()
plt.show()

df = pd.read_csv("C:/Users/vikto/Downloads/dataset_209770_6 (3).txt", sep=" ")
sns.scatterplot(x="x", y="y", data = df)
plt.show()

df = pd.read_csv("C:/Users/vikto/Downloads/genome_matrix.csv", index_col=0)
# Визуализация heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f")
plt.title("Корреляция признаков")
plt.show()

dota2 = pd.read_csv("C:/Users/vikto/Downloads/dota_hero_stats.csv")
dota2["count"]= dota2.roles.str.split().str.len()
sns.histplot(dota2, x = "count")
plt.show()

flower = pd.read_csv("C:/Users/vikto/Downloads/iris.csv")
for column in df:
    sns.kdeplot(data=flower)
    plt.xlim([-2.5,10])
    plt.show()

flower = pd.read_csv("C:/Users/vikto/Downloads/iris.csv")
sns.violinplot(flower, y = "petal length")
plt.ylim([-4,10])
plt.show()

flower = pd.read_csv("C:/Users/vikto/Downloads/iris.csv")
sns.pairplot(flower)
plt.show()

#PROJECT ONE
event_data_train = pd.read_csv("C:/Users/vikto/Downloads/event_data_train/event_data_train.csv")
event_data_train["time"] = pd.to_datetime(event_data_train.timestamp, unit="s")
event_data_train["day"] = event_data_train["time"].dt.date
event_data_train.head()

event_data = event_data_train.pivot_table(index = "user_id",
                             columns = "action", 
                             values = "step_id", 
                             aggfunc = "count",
                             fill_value = 0).reset_index()
event_data.head()

#ИЩУ ID АВТОРА
submissions_data = pd.read_csv("C:/Users/vikto/Downloads/event_data_train/submissions_data_train.csv")
submissions_data.head()
submissions_data.submission_status.unique()
submissions_data = submissions_data \
    .groupby(["user_id","submission_status"]) \
        .aggregate({"submission_status":"count"})

submissions_data = submissions_data \
    .groupby(["user_id"]).sum()

submissions_data = submissions_data.sort_values(["submission_status"], ascending = False)
#У АВТОРА КУРСА ID 1046 )))

#ЛАДНО, ПОЕХАЛИ ДАЛЬШЕ
user_score = submissions_data.pivot_table(index = "user_id",
                             columns = "submission_status", 
                             values = "step_id", 
                             aggfunc = "count",
                             fill_value = 0).reset_index()
user_score.head()
gaps_data = event_data_train[['user_id', "timestamp", "day"]].drop_duplicates(subset = "day")\
    .groupby("user_id")["timestamp"].apply(list).apply(np.diff).values
gaps_data = pd.Series(np.concatenate(gaps_data, axis=0))
gaps_data.hist(gaps_data < 200)
plt.show()

month = 30 * 24 * 60 * 60
now = submissions_data.timestamp.max()
gaps_data.quantile(0.90) / (24 * 60 * 60 )
users_data = event_data_train.groupby("user_id", as_index = False).agg({"timestamp": "max"})\
    .rename(columns= {"timestamp":"last_timestamp"})
users_data["is_gone"] = (now - users_data.last_timestamp)> month
users_data.head()
users_data = users_data.merge(user_score,on = "user_id", how = "outer")
users_data = users_data.fillna(0)
users_data.head()
users_days = event_data_train.groupby("user_id").day.nunique().to_frame().reset_index()
users_days.head()
users_data = users_data.merge(event_data, on = "user_id", how = "outer")
users_data = users_data.merge(users_days, on = "user_id", how = "outer")
users_data.user_id.nunique()
users_data["passed_course"] = users_data.passed > 170

#ПРОДОЛЖЕНИЕ
user_min_time = event_data_train.groupby("user_id", as_index = False).agg({"timestamp": "min"}).rename({"timestamp": "min_timestamp"}, axis = 1)


event_data_train = event_data_train.merge(user_min_time, how = 'outer')
events_data_3_days = event_data_train.query("timestamp <= min_timestamp + 3 * 24 * 60* 60")
events_data_3_days.groupby("user_id").day.nunique().max()

submissions_data_user_min_time = submissions_data.groupby("user_id", as_index = False).agg({"timestamp": "min"}).rename({"timestamp": "min_timestamp"}, axis = 1)
submissions_data = submissions_data.merge(submissions_data_user_min_time, how = 'outer')
submissions_data_train = submissions_data.query("timestamp <= min_timestamp + 3 * 24 * 60* 60")

#ЧУЖОЕ ДОБРО
import pandas as pd
import numpy as np 

%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns

events_data = pd.read_csv('event_data_train.csv')
submissions_data = pd.read_csv('submissions_data_train.csv')
events_data['date'] = pd.to_datetime(events_data.timestamp, unit = 's')
events_data['day'] = events_data.date.dt.date

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit = 's')
submissions_data['day'] = submissions_data.date.dt.date

users_data = events_data.groupby('user_id', as_index= False)\
        .agg({'timestamp': 'max'})\
        .rename({'timestamp':'last_timestamp'}, axis = 'columns')
now = 1526772811
drop_out_threasold = 2592000

users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threasold
users_scores = submissions_data.pivot_table(index='user_id',\
                        columns = 'submission_status',\
                        values = 'step_id',\
                        aggfunc = 'count',\
                        fill_value = 0)\
                        .reset_index()
users_data = users_data.merge(users_scores, on = 'user_id', how = 'outer')
users_data = users_data.fillna(0)
users_invent_data = events_data.pivot_table(index = 'user_id',
                        columns = 'action',
                        values  = 'step_id',
                        aggfunc = 'count',
                        fill_value = 0).reset_index()
users_data = users_data.merge(users_invent_data, how = 'outer')
users_days = events_data.groupby('user_id').day.nunique()
users_days.to_frame().reset_index()
users_data = users_data.merge(users_days, on = 'user_id', how = 'outer')
users_data['passed_corse'] = users_data.passed > 170

submissions_data.step_id.mode()

submissions_data

events_data

submissions_data[submissions_data.submission_status=='wrong']\
.groupby("step_id").agg({'submission_status': 'count'}).\
sort_values('submission_status', ascending=True).head(10)

user_min_time= events_data.groupby('user_id', as_index= False).\
agg({'timestamp':'min'})\
.rename({"timestamp":'min_timestamp'}, axis=1)
users_data = users_data.merge(user_min_time, how = 'outer')
users_data

events_data

event_data_train = pd.DataFrame()

events_data['user_time']=events_data.user_id.map(str)\
+ "_" + events_data.timestamp.map(str)

events_data

learning_time_threshold=3*24*60*60

user_learning_time_threshold=user_min_time.user_id.map(str)\
+'_'+(learning_time_threshold + user_min_time.min_timestamp).map(str)

user_min_time["user_learning_time_threshold"] = user_learning_time_threshold
user_min_time

events_data=events_data\
.merge(user_min_time[['user_id', 'user_learning_time_threshold']],
       how = 'outer')

events_data.shape

events_data_train= events_data[events_data.user_time<=events_data.user_learning_time_threshold]

events_data_train.groupby('user_id').day.nunique().max()

submissions_data['users_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'user_learning_time_threshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.users_time <= submissions_data.user_learning_time_threshold]
submissions_data_train.groupby('user_id').day.nunique().max()

X=submissions_data_train.groupby('user_id').day\
.nunique().to_frame().reset_index().rename(columns={'day':'days'})

steps_tried=submissions_data_train.groupby('user_id').step_id.nunique()\
.to_frame().reset_index().rename(columns={'step_id':'steps_tried'})

X=X.merge(steps_tried, on='user_id', how = 'outer')

X=X.merge(submissions_data_train.pivot_table(index='user_id',\
                        columns = 'submission_status',\
                        values = 'step_id',\
                        aggfunc = 'count',\
                        fill_value = 0)\
                        .reset_index())

X['correct_ratio'] = X.correct / (X.correct+X.wrong)

X

X=X.merge(events_data_train.pivot_table(index='user_id',
                        columns = 'action',\
                        values = 'step_id',\
                        aggfunc = 'count',\
                        fill_value = 0)\
                        .reset_index()[['user_id', 'viewed']], how = 'outer')

X=X.fillna(0)
users_data

X=X.merge(users_data[['user_id', 'passed_corse', 'is_gone_user']], how = 'outer')

X

X=X[~((X.is_gone_user==False)&(X.passed_corse==False))]

X.groupby(['passed_corse', 'is_gone_user']).user_id.count()

y=X.passed_corse.map(int)

X=X.drop(['passed_corse', 'is_gone_user'], axis=1)

X=X.set_index(X.user_id)
X=X.drop('user_id', axis=1)

X

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
dt = DecisionTreeClassifier(criterion='entropy')
parameters = {'max_depth': range(3,6), 'max_leaf_nodes':range(6,15), 'min_samples_leaf': range(1,4),'min_samples_split':range(2,5)}
grid_search_cv_clf = GridSearchCV(dt,parameters,cv=4)
grid_search_cv_clf.fit(X_train,y_train)
model = grid_search_cv_clf.best_estimator_
print(grid_search_cv_clf.best_params_, cross_val_score(model,X_train,y_train,cv=4).mean())

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(15,25), 'leaf_size':range(1,7)}
grid_search_cv_clf = GridSearchCV(knn,parameters,cv=4,n_jobs=-1)
grid_search_cv_clf.fit(X_train,y_train)
model = grid_search_cv_clf.best_estimator_
print(grid_search_cv_clf.best_params_, cross_val_score(model,X_train,y_train,cv=4).mean())




