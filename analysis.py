#load libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='whitegrid', palette='pastel')
import matplotlib.pyplot as plt
from functools import reduce
import sklearn

#load tables
copa19 = pd.read_csv('data_tables\copa19.csv')
copa21 = pd.read_csv('data_tables\copa21.csv')
euro21 = pd.read_csv('data_tables\euro21.csv')
worldc18 = pd.read_csv('data_tables\worldc18.csv')
worldc22 = pd.read_csv('data_tables\worldc22.csv')


#merge competition dfs
comps = pd.concat([copa19, copa21, euro21, worldc18, worldc22])
comps = comps.fillna(0)
comps = comps.reset_index(drop=True)
comps = comps.rename({'Unnamed: 0': 'Team_Game'}, axis=1)
comps

#EDA
#show correlation plot across all features
sns.heatmap(comps.iloc[:, 1:].corr(), vmin=-1, vmax=1, annot=True, annot_kws={"fontsize":6})
plt.show()

#show correlation with result variable
sns.heatmap(comps.iloc[:, 1:].corr()[['Result']].sort_values(by='Result', ascending=False), vmin=-1, vmax=1, annot=True, annot_kws={"fontsize":6})
plt.show()


#load statistical analysis libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#create training and testing dataframes
y_value = comps.iloc[:, -1:]
x_cols = comps.iloc[:, 1:-1]
X_train, X_test, y_train, y_test = train_test_split(x_cols, y_value, test_size=0.25, random_state=13)

#check for multicollinearity
x_vif = pd.DataFrame({'Features': x_cols.columns, 'VIF' :[variance_inflation_factor(x_cols, i) for i in range(len(x_cols.columns))]})
x_vif 
###all but two variables are strongly multicollinear - we shall implement a tree-based classification model which
###are unaffected by highly correlated data as they perform implicit feature selection during the training process
###https://www.linkedin.com/pulse/need-check-multicollinearity-remove-correlated-variables-sen/
###https://arxiv.org/pdf/2111.02513

#random forest approach
rfc = RandomForestClassifier(n_estimators= 100).fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

metrics.accuracy_score(y_test, rfc_y_pred) #73.50
pd.Series(rfc.feature_importances_, index=x_cols.columns).sort_values(ascending=False)
print(classification_report(y_test, rfc_y_pred)) #83% precision for Win classification

#hypertune with grid search
param_grid = { 
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid) 
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_) #(max_depth=6, max_features=None, max_leaf_nodes=9, n_estimators=25

rfc_hypergrid = RandomForestClassifier(max_depth=6, max_features=None, max_leaf_nodes=9, n_estimators=25).fit(X_train, y_train)
rfc_hypergrid_pred = rfc_hypergrid.predict(X_test)

metrics.accuracy_score(y_test, rfc_hypergrid_pred) #70.09
pd.Series(rfc_hypergrid.feature_importances_, index=x_cols.columns).sort_values(ascending=False)
print(classification_report(y_test, rfc_hypergrid_pred)) #80% precision for Win classification

#hypertune with random search
random_search = RandomizedSearchCV(RandomForestClassifier(), param_grid)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_) #max_depth=3, max_features=None, max_leaf_nodes=9, n_estimators=50

rfc_hyperrand = RandomForestClassifier(max_depth=3, max_features=None, max_leaf_nodes=9, n_estimators=50).fit(X_train, y_train)
rfc_hyperrand_pred = rfc_hyperrand.predict(X_test)

metrics.accuracy_score(y_test, rfc_hyperrand_pred) #71.79
pd.Series(rfc_hyperrand.feature_importances_, index=x_cols.columns).sort_values(ascending=False)
print(classification_report(y_test, rfc_hyperrand_pred)) #77% precision for Win classification

#xgboost random forest approach
import xgboost
from xgboost import XGBRFClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_value_xgb = le.fit_transform(y_value)
y_train_xgb = le.fit_transform(y_train)
y_test_xgb = le.fit_transform(y_test)


xgbrfc = XGBRFClassifier(n_estimators=100, random_state=13).fit(X_train, y_train_xgb)
xgbrfc_pred = xgbrfc.predict(X_test)
metrics.accuracy_score(y_test_xgb, xgbrfc_pred) #76.92
print(classification_report(y_test_xgb, xgbrfc_pred)) #84% precision for Win classification
#pd.Series(xgbrfc.feature_importances_, index=x_cols.columns).sort_values(ascending=False)

xgbrf_features = pd.DataFrame(pd.Series(xgbrfc.feature_importances_, index=x_cols.columns))
xgbrf_features = xgbrf_features.rename(columns = {0 : "Feature_Importance"})
xgbrf_features.sort_values(by="Feature_Importance", ascending=False)

sns.heatmap((xgbrf_features.sort_values(by="Feature_Importance", ascending=False)), vmin=0, vmax=0.15, annot=True, annot_kws={"fontsize":6})
plt.show()

###from initial performance, the xgboost random forest approach will return the best accuracy and precision for
###wins, so we shall proceed with this model. In the next step, we remove the goals feature as it is redundant 
###for our modelling problem (if you score more goals, you're more likely to win -> we want to more clearly define
###why teams that win are able to score more goals, or win these games)

#remove goals from x_features
X_traing = X_train.drop('Goals', axis=1)
X_testg = X_test.drop('Goals', axis=1)

xgbrfc = XGBRFClassifier(n_estimators=100, random_state = 13).fit(X_traing, y_train_xgb)
xgbrfc_pred = xgbrfc.predict(X_testg)
metrics.accuracy_score(y_test_xgb, xgbrfc_pred) #72.65
print(classification_report(y_test_xgb, xgbrfc_pred)) #83% precision for Win classification
pd.Series(xgbrfc.feature_importances_, index=x_cols.columns.drop('Goals')).sort_values(ascending=False)

xgbrf_features = pd.DataFrame(pd.Series(xgbrfc.feature_importances_, index=x_cols.columns.drop('Goals')))
xgbrf_features = xgbrf_features.rename(columns = {0 : "Feature_Importance"})
xgbrf_features.sort_values(by="Feature_Importance", ascending=False)

sns.heatmap((xgbrf_features.sort_values(by="Feature_Importance", ascending=False)), vmin=0, vmax=0.15, annot=True, annot_kws={"fontsize":6})
plt.show()
###we now see that assists are the most important feature - but there are still over 30 that we use, so we next aim
###to optimize our feature count using the fisher_score approach

#feature selection
from skfeature.function.similarity_based import fisher_score

fishy = fisher_score.fisher_score(np.array(X_traing), y_train_xgb, mode='rank')
ft_importance = pd.Series(fishy, X_traing.columns)
ft_importance = pd.DataFrame(ft_importance).reset_index().rename({'index': 'Features', 0: 'Fisher Rank'}, axis=1)
ft_importance = ft_importance.sort_values(by=["Fisher Rank"], ascending=False)

sns.barplot(ft_importance, x="Features", y="Fisher Rank")
plt.xticks(rotation=80)
plt.show()

xgb_scores = []
for i in range(1, 31):
    xgbrfc_ft = XGBRFClassifier(n_estimators=100).fit(X_traing[ft_importance.iloc[0:i]['Features']], y_train_xgb)
    xgbrfc_ft_pred = xgbrfc_ft.predict(X_testg[ft_importance.iloc[0:i]['Features']])
    xgb_scores.append(metrics.accuracy_score(y_test_xgb, xgbrfc_ft_pred));

xgb_ftscore = pd.DataFrame({"# Features": list(range(1, 31)), 'xgb_score': xgb_scores})
xgb_ftscore #29 has max score of 74.3590; other 25-30 values have score of 72.6496

xgb_ftplot = sns.barplot(xgb_ftscore, x="# Features", y="xgb_score")
plt.xticks(rotation=80)
for i in xgb_ftplot.containers:
    xgb_ftplot.bar_label(i,);
plt.show()

param_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 350],
    'max_depth': [0, 2, 4, 6, 8, 10],
    'learning_rate': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'subsample': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'colsample_bynode': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
}

#hypertune with random search
estimators = []
for i in range(25, 31):
    random_search = RandomizedSearchCV(XGBRFClassifier(random_state=13), param_grid)
    random_search.fit(X_traing[ft_importance.iloc[0:i]['Features']], y_train_xgb)
    estimators.append(random_search.best_estimator_);

est25 = str(estimators[0])
est25[est25.find('colsample_bynode'): est25.find('colsample_bynode') + len('colsample_bynode') + 4] #0.6
est25[est25.find('learning_rate'): est25.find('learning_rate') + len('learning_rate') + 4] #0.4
est25[est25.find('max_depth'): est25.find('max_depth') + len('max_depth') + 4] #0
est25[est25.find('n_estimators'): est25.find('n_estimators') + len('n_estimators') + 4] #150
est25[est25.find('subsample'): est25.find('subsample') + len('subsample') + 4] #N/A

est26 = str(estimators[1])
est26[est26.find('colsample_bynode'): est26.find('colsample_bynode') + len('colsample_bynode') + 4] #0.6
est26[est26.find('learning_rate'): est26.find('learning_rate') + len('learning_rate') + 4] #NA
est26[est26.find('max_depth'): est26.find('max_depth') + len('max_depth') + 4] #4
est26[est26.find('n_estimators'): est26.find('n_estimators') + len('n_estimators') + 4] #350
est26[est26.find('subsample'): est26.find('subsample') + len('subsample') + 4] #N/A

est27 = str(estimators[2])
est27[est27.find('colsample_bynode'): est27.find('colsample_bynode') + len('colsample_bynode') + 4] #0.2
est27[est27.find('learning_rate'): est27.find('learning_rate') + len('learning_rate') + 4] #0.4
est27[est27.find('max_depth'): est27.find('max_depth') + len('max_depth') + 4] #0
est27[est27.find('n_estimators'): est27.find('n_estimators') + len('n_estimators') + 4] #150
est27[est27.find('subsample'): est27.find('subsample') + len('subsample') + 4] #N/A

est28 = str(estimators[3])
est28[est28.find('colsample_bynode'): est28.find('colsample_bynode') + len('colsample_bynode') + 4] #0.4
est28[est28.find('learning_rate'): est28.find('learning_rate') + len('learning_rate') + 4] #0.8
est28[est28.find('max_depth'): est28.find('max_depth') + len('max_depth') + 4] #8
est28[est28.find('n_estimators'): est28.find('n_estimators') + len('n_estimators') + 4] #300
est28[est28.find('subsample'): est28.find('subsample') + len('subsample') + 4] #N/A

est29 = str(estimators[4])
est29[est29.find('colsample_bynode'): est29.find('colsample_bynode') + len('colsample_bynode') + 4] #0.2
est29[est29.find('learning_rate'): est29.find('learning_rate') + len('learning_rate') + 4] #0.8
est29[est29.find('max_depth'): est29.find('max_depth') + len('max_depth') + 4] #8
est29[est29.find('n_estimators'): est29.find('n_estimators') + len('n_estimators') + 4] #250
est29[est29.find('subsample'): est29.find('subsample') + len('subsample') + 4] #N/A

est30 = str(estimators[5])
est30[est30.find('colsample_bynode'): est30.find('colsample_bynode') + len('colsample_bynode') + 4] #NA
est30[est30.find('learning_rate'): est30.find('learning_rate') + len('learning_rate') + 4] #0.2
est30[est30.find('max_depth'): est30.find('max_depth') + len('max_depth') + 4] #0
est30[est30.find('n_estimators'): est30.find('n_estimators') + len('n_estimators') + 4] #150
est30[est30.find('subsample'): est30.find('subsample') + len('subsample') + 4] #N/A

#25: colsample_bynode= 0.6, learning_rate= 0.4, max_depth= 0, n_estimators= 150
#26: colsample_bynode= 0.6, learning_rate= ___, max_depth= 4, n_estimators= 350
#27: colsample_bynode= 0.2, learning_rate= 0.4, max_depth= 0, n_estimators= 150
#28: colsample_bynode= 0.4, learning_rate= 0.8, max_depth= 8, n_estimators= 300
#29: colsample_bynode= 0.2, learning_rate= 0.8, max_depth= 8, n_estimators= 250
#30: colsample_bynode= ___, learning_rate= 0.2, max_depth= 0, n_estimators= 150

#try suggested best estimators for feature counts 25-30
xgbrfc_25 = XGBRFClassifier(colsample_bynode= 0.6, learning_rate= 0.4, max_depth= 0, n_estimators= 150, random_state=13).fit(X_traing[ft_importance.iloc[0:25]['Features']], y_train_xgb)
xgbrfc_25_pred = xgbrfc_25.predict(X_testg[ft_importance.iloc[0:25]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_25_pred) #70.94
print(classification_report(y_test_xgb, xgbrfc_25_pred)) #82% precision for Win classification

xgbrfc_26 = XGBRFClassifier(colsample_bynode= 0.6, max_depth= 4, n_estimators= 350, random_state=13).fit(X_traing[ft_importance.iloc[0:26]['Features']], y_train_xgb)
xgbrfc_26_pred = xgbrfc_26.predict(X_testg[ft_importance.iloc[0:26]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_26_pred) #74.36
print(classification_report(y_test_xgb, xgbrfc_26_pred)) #84% precision for Win classification

xgbrfc_27 = XGBRFClassifier(colsample_bynode= 0.2, learning_rate= 0.4, max_depth= 0, n_estimators= 150, random_state=13).fit(X_traing[ft_importance.iloc[0:27]['Features']], y_train_xgb)
xgbrfc_27_pred = xgbrfc_27.predict(X_testg[ft_importance.iloc[0:27]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_27_pred) #70.94
print(classification_report(y_test_xgb, xgbrfc_27_pred)) #76% precision for Win classification

xgbrfc_28 = XGBRFClassifier(colsample_bynode= 0.4, learning_rate= 0.8, max_depth= 8, n_estimators= 300, random_state=13).fit(X_traing[ft_importance.iloc[0:28]['Features']], y_train_xgb)
xgbrfc_28_pred = xgbrfc_28.predict(X_testg[ft_importance.iloc[0:28]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_28_pred) #70.94
print(classification_report(y_test_xgb, xgbrfc_28_pred)) #80% precision for Win classification

xgbrfc_29 = XGBRFClassifier(colsample_bynode= 0.2, learning_rate= 0.8, max_depth= 8, n_estimators= 250, random_state=13).fit(X_traing[ft_importance.iloc[0:29]['Features']], y_train_xgb)
xgbrfc_29_pred = xgbrfc_29.predict(X_testg[ft_importance.iloc[0:29]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_29_pred) #71.79
print(classification_report(y_test_xgb, xgbrfc_29_pred)) #78% precision for Win classification

xgbrfc_30 = XGBRFClassifier(learning_rate= 0.2, max_depth= 0, n_estimators= 150, random_state=13).fit(X_traing[ft_importance.iloc[0:30]['Features']], y_train_xgb)
xgbrfc_30_pred = xgbrfc_30.predict(X_testg[ft_importance.iloc[0:30]['Features']])
metrics.accuracy_score(y_test_xgb, xgbrfc_30_pred) #71.79
print(classification_report(y_test_xgb, xgbrfc_30_pred)) #83% precision for Win classification

###select xgbrfc_26

pd.Series(xgbrfc_26.feature_importances_, index=X_traing[ft_importance.iloc[0:26]['Features']].columns)

xgbrf_features = pd.DataFrame(pd.Series(xgbrfc_26.feature_importances_, index=X_traing[ft_importance.iloc[0:26]['Features']].columns))
xgbrf_features = xgbrf_features.rename(columns = {0 : "Feature_Importance"})
xgbrf_features.sort_values(by="Feature_Importance", ascending=False)

#show ten most important features
sns.heatmap((xgbrf_features.sort_values(by="Feature_Importance", ascending=False).iloc[0:11]), vmin=0, vmax=0.2, annot=True, annot_kws={"fontsize":6})
plt.show()