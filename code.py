# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Code starts here
df=pd.read_csv(filepath_or_buffer=path,compression='zip',low_memory=False)

X=df.drop(columns=["loan_status"],axis=1)
y=df["loan_status"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=4)
# Code ends here


# --------------
# Code starts  here
col=df.isnull().sum()
column_list=df.columns
col_drop=[]
for column in column_list:
    if df[column].isnull().sum()/len(df[column])*100 > 25:
        col_drop.append(column)        

for column in column_list:
    if df[column].nunique()==1:
        col_drop.append(column)

X_train=X_train.drop(columns=col_drop,axis=1)
X_test=X_test.drop(columns=col_drop,axis=1)


# Code ends here


# --------------
import numpy as np


# Code starts here
y_train=np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)
y_test=np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Code starts here
for column in cat:
    X_train[column]=X_train[column].fillna(X_train[column].mode()[0])
    X_test[column]=X_test[column].fillna(X_test[column].mode()[0])

for column in num:
    X_train[column]=X_train[column].fillna(X_train[column].mean())
    X_test[column]=X_test[column].fillna(X_test[column].mean())

le=LabelEncoder()
for col in cat:
    X_train[col]=le.fit_transform(X_train[col])
    X_test[col]=le.fit_transform(X_test[col])

# Code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# rf = RandomForestClassifier()

rf = RandomForestClassifier(random_state= 42,max_depth=2,min_samples_leaf=5000)

rf.fit(X_train,y_train)

accuracy = rf.score(X_test,y_test)

y_pred = rf.predict(X_test)

# Store the different evaluation values.

f1 = f1_score(y_test, rf.predict(X_test))
precision = precision_score(y_test, rf.predict(X_test))
recall = recall_score(y_test, rf.predict(X_test))
roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



# Plot the auc-roc curve

score = roc_auc_score(y_pred , y_test)
y_pred_proba = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Random Forrest, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# --------------
from xgboost import XGBClassifier

# Code starts here
# rf = RandomForestClassifier()

xgb = XGBClassifier(learning_rate=0.0001)

xgb.fit(X_train,y_train)

accuracy = xgb.score(X_test,y_test)

y_pred = xgb.predict(X_test)

# Store the different evaluation values.

f1 = f1_score(y_test, xgb.predict(X_test))
precision = precision_score(y_test, xgb.predict(X_test))
recall = recall_score(y_test, xgb.predict(X_test))
roc_auc = roc_auc_score(y_test, xgb.predict(X_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



# Plot the auc-roc curve

score = roc_auc_score(y_pred , y_test)
y_pred_proba = xgb.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Random Forrest, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

# Code ends here


